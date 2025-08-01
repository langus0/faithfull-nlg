import argparse
import json
import os
import asyncio
import time
from pathlib import Path
from jinja2 import Template
from typing import Callable

from loguru import logger

from mods import (
    modify_text_severity,
    modify_severity, 
    strip_forbidden_symbols,
    modify_add_critical_error,
    modify_add_random_error
)
from vllm import LLM, SamplingParams

class VLLMLM():
#mbley/google-gemma-2-27b-it-AWQ
    def __init__(self, model_name="mbley/google-gemma-2-27b-it-AWQ"):
        self.pending_chats = []
        
    def init_model(self, model_name="mbley/google-gemma-2-27b-it-AWQ"):
        if model_name == "eval_gemma":
            model_name = "mbley/google-gemma-2-27b-it-AWQ"
        elif model_name == "eval_nemo":
            model_name = "joshmiller656/Llama-3.1-Nemotron-70B-Instruct-AWQ-INT4"
        elif model_name == "eval_qwen":
            model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"
        self.llm = LLM(model=model_name,  max_model_len=4000,
                       gpu_memory_utilization=0.80, max_num_seqs=100, tensor_parallel_size=2,enable_prefix_caching=True) 


    # Chat function waits for someone to provide the response later
    async def chat(self, model, messages):
        future = asyncio.Future()
        my_messages = [{"role": "system", "content": "You are an expert evaluator of AI models with extraordinary skills in identifying and analyzing errors in model outputs."}]
        my_messages.extend(messages)
        self.pending_chats.append((my_messages, future))
        return await future
    
    def execute(self):
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=2000,repetition_penalty = 1.1, top_k=40, top_p=0.9)   
        print("Executing LLM...")
        print(len(self.pending_chats), "pending chats")
        prompts = [text for text, future in self.pending_chats]
        outputs = self.llm.chat(prompts, sampling_params, add_generation_prompt=False, continue_final_message= True)
        outputs = [output.outputs[0].text for output in outputs]
        for (text, future), output in zip(self.pending_chats, outputs):
            response = {}
            response['message'] = {}
            response['message']['content'] = output
            future.set_result(response)
        self.pending_chats.clear()
        return outputs  
    
class FakeVLLMLM():
#mbley/google-gemma-2-27b-it-AWQ
    def __init__(self, model_name="mbley/google-gemma-2-27b-it-AWQ"):
        self.pending_chats = []
        

    # Chat function waits for someone to provide the response later
    async def chat(self, model, messages):
        future = asyncio.Future()
        my_messages = [{"role": "system", "content": "You are an expert evaluator of AI models with extraordinary skills in identifying and analyzing errors in model outputs."}]
        my_messages.extend(messages)
        self.pending_chats.append((my_messages, future))
        return await future
    
    def execute(self):
        print("Executing LLM...")
        print(len(self.pending_chats), "pending chats")
        for text, future in self.pending_chats:
            response = {}
            response['message'] = {}
            response['message']['content'] = "SUPER!"
            future.set_result(response)
        self.pending_chats.clear()


lm = VLLMLM()  # Initialize the LLM instance

async def empty_modify(result: str, model: str, mod_force: int, example:dict, lm: VLLMLM) -> str:
    """
    A dummy modification function that does nothing.
    """
    return None

EVAL_MODS = {
    "severity": modify_severity,
    "text_severity": modify_text_severity,
    "add_critical_error": modify_add_critical_error,
    "add_random_error": modify_add_random_error,
    "none": empty_modify
}

async def modify_result(
        prompt: str,
        result: str,
        model: str,
        eval_mod: str,
        mod_force: int,
        example: dict
    ) -> str:
    """
    Parse the result from the model based on the evaluation modification.
    This function can use different functions for error modification.
    """
    
    if eval_mod in ["add_random_error"]:
        example['prompt'] = prompt
    
    modified_result = await EVAL_MODS[eval_mod](result, model, mod_force, example, lm)
    if modified_result:
        response = await lm.chat(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': modified_result}
            ]
        )
        return modified_result + strip_forbidden_symbols(response['message']['content'])
    else:
        return None
    

async def run_evaluation(
        template: Template,
        aspect_config: dict,
        data: dict,
        model: str,
        output_dir: str,
        skip: int,
        limit: int,
        eval_mod: str,
        mod_force: int,
        use_premodified_result: bool,
    ):

    data = data[skip:limit + skip if limit else None]
    tasks = []    
    print("Adding tasks for evaluation...")
    for i, example in enumerate(data):
        tasks.append(asyncio.create_task(process_example(template, aspect_config, model, output_dir, eval_mod, mod_force, use_premodified_result, i, example)))
        
    # Wait a moment to simulate async task startup
    await asyncio.sleep(1)
    while len(lm.pending_chats) != 0:
        lm.execute()
        await asyncio.sleep(2)
    print("All LLMs executions performed, now waiting for tasks to finish...")
    await asyncio.gather(*tasks)

async def process_example(template, aspect_config, model, output_dir, eval_mod, mod_force, use_premodified_result, i, example):
    prompt = template.render(
                inputs=example['inputs'],
                outputs=example['outputs'],
                **aspect_config
            )
        
        # when the original dataset is used, it has only inputs and outputs, and the result should be generated
        # if a pregen file is used, then it should contain the result already and the generation can be skipped
    result = example.get('result', None)
    if result is None:
        logger.info(f"Example {example['id']} has no result, generating.")
        response = await lm.chat(model=model, messages=[{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': ""}])
        result = response['message']['content']
        result = strip_forbidden_symbols(result)

    output_path = Path(output_dir) / f'{example["id"]}.json'
    eval_output = {
            'inputs': example['inputs'],
            'outputs': example['outputs'],
            'result': result
        }
        
        # a pregen file might also contain results from previous modifications
        # (this should be described in the name of the pregen file)
    result_to_modify = example.get('result_premodified', result) if use_premodified_result else result

    if eval_mod:
        modified_result = await modify_result(prompt, result_to_modify, model, eval_mod, mod_force, example)
        if modified_result:
            eval_output['result_modified'] = modified_result
        else:
            eval_output['result_modified'] = result # No modification was needed
            logger.info(f"Example {example['id']} had no modification needed, continuing with original result.")

    logger.info(f"\n[{i}] Saving {example['id']} to {output_path}\n")
    #logger.info(eval_output)
    with open(output_path, 'w') as f:
        
        f.write(json.dumps(eval_output, indent=2))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, help='Path to the prompt template file')
    parser.add_argument('--aspect-config', type=str, help='Path to the aspect configuration file')
    parser.add_argument('--data', type=str, help='Path to the dataset JSON inputs and outputs to evaluate, or a pregen JSON with pregenerated model results')
    parser.add_argument('--model', type=str, default='none', help='Ollama model name')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--skip', type=int, default=0, help='Slice of examples to evaluate if there should be less than all')
    parser.add_argument('--limit', type=int, default=None, help='Slice of examples to evaluate if there should be less than all')
    parser.add_argument('--eval-mod', type=str, default=None, help='Modification to apply to the evaluation process')
    parser.add_argument('--mod-force', type=int, help='Force of the severity modification, negative for less severity, positive for more severity')
    parser.add_argument('--use-premodified-result', action='store_true', help='Use the pre-modified result from the pregen if available')
    args = parser.parse_args()

    template_path = Path(args.template)
    aspect_config_path = Path(args.aspect_config)
    data_path = Path(args.data)

    with open(template_path, 'r') as f:
        template = Template(f.read())

    with open(aspect_config_path, 'r') as f:
        aspect_config = json.load(f)

    with open(data_path, 'r') as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()
    
    lm.init_model(args.model)
    
    try:
        asyncio.run(run_evaluation(
            template,
            aspect_config,
            data,
            args.model,
            args.output_dir,
            args.skip,
            args.limit,
            args.eval_mod,
            args.mod_force,
            args.use_premodified_result
        ))
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        raise e

    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")
