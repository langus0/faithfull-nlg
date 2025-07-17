import argparse
import json
import os
import asyncio
import time
from pathlib import Path
from jinja2 import Template
from typing import Callable

#from ollama import chat
from loguru import logger

from mods_vllm import modify_impact_per_error, modify_delete_per_error, strip_forbidden_symbols
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

lm = VLLMLM()  # Initialize the LLM instance

async def empty_modify(prompt: str, result: str, model: str, mod_direction: int) -> str:
    """
    A dummy modification function that does nothing.
    """
    return None

EVAL_MODS = {
    "impact": modify_impact_per_error,
    "delete": modify_delete_per_error,
    "none": empty_modify
}

async def run_evaluation(
        template: Template,
        aspect_config: dict,
        data: dict,
        model: str,
        output_dir: str,
        skip: int,
        limit: int,
        per_error_mod: str,
        mod_direction: int,
    ):

    data = data[skip:limit + skip if limit else None]
    
    tasks = []    
    print("Adding tasks for evaluation...")
    for i, example in enumerate(data):
        tasks.append(asyncio.create_task(process_example(
            template,
            aspect_config,
            model,
            output_dir,
            per_error_mod,
            mod_direction,
            i,
            example
        ))
    )
    # Wait a moment to simulate async task startup
    await asyncio.sleep(1)
    while len(lm.pending_chats) != 0:
        lm.execute()
        await asyncio.sleep(2)
    print("All LLMs executions performed, now waiting for tasks to finish...")
    await asyncio.gather(*tasks)
    
    ###################
    
async def process_example(
        template: Template,
        aspect_config: dict,
        model: str,
        output_dir: str,
        per_error_mod: str,
        mod_direction: int,
        i: int,
        example: dict):
    """
    Process a single example for evaluation.
    This includes multiple LLM calls.
    """
    
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
        response = await lm.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        result = response['message']['content']
        result = strip_forbidden_symbols(result)

    output_path = Path(output_dir) / f'{example["id"]}.json'
    eval_output = {
        'result': result,
        'error_mod_impacts': []
    }

    error_mod_impacts = await EVAL_MODS[per_error_mod](prompt, result, model, lm, mod_direction)

    if error_mod_impacts:
        eval_output['error_mod_impacts'] = error_mod_impacts
    else:
        eval_output['error_mod_impacts'] = None # No error modification was applied
        logger.info(f"Example {example['id']} had no errors to modify.")

    logger.info(f"\n[{i}] Saving {example['id']} to {output_path}\n")
    with open(output_path, 'w') as f:
        f.write(json.dumps(eval_output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, help='Path to the prompt template file')
    parser.add_argument('--aspect-config', type=str, help='Path to the aspect configuration file')
    parser.add_argument('--data', type=str, help='Path to the dataset JSON inputs and outputs to evaluate, or a pregen JSON with pregenerated model results')
    parser.add_argument('--model', type=str, default='eval_nemo', help='Ollama model name')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--skip', type=int, default=0, help='Slice of examples to evaluate if there should be less than all')
    parser.add_argument('--limit', type=int, default=None, help='Slice of examples to evaluate if there should be less than all')
    parser.add_argument('--per_error_mod', type=str, default='none', help='Modification to apply per error severity')
    parser.add_argument('--mod-direction', type=int, help='Force of the severity modification, either +1 or -1 for increasing or decreasing severity')
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
            args.per_error_mod,
            args.mod_direction,
        ))
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        raise e

    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")
