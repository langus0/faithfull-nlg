import argparse
import json
import os
import time
from pathlib import Path
from jinja2 import Template
from typing import Callable

from ollama import chat
from loguru import logger

from mods import modify_text_severity, modify_severity

EVAL_MODS = {
    "severity": modify_severity,
    "text_severity": modify_text_severity
}

def modify_result(
        prompt: str,
        result: str,
        model: str,
        eval_mod: str,
        mod_force: int
    ) -> str:
    """
    Parse the result from the model based on the evaluation modification.
    This function can use different functions for error modification.
    """
    
    modified_result = EVAL_MODS[eval_mod](result, model, mod_force)

    if modified_result:
        response = chat(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': modified_result}
            ]
        )
        return modified_result + response['message']['content']
    else:
        return None

def run_evaluation(
        template: Template,
        aspect_config: dict,
        data: dict, model: str,
        output_dir: str,
        skip: int,
        limit: int,
        eval_mod: str,
        mod_force: int
    ):

    data = data[skip:limit + skip if limit else None]
    
    for i, example in enumerate(data):
        
        result = example.get('result', None)
        if result is None:
            logger.info(f"Example {example['id']} has no result, generating.")
            prompt = template.render(
                inputs=example['inputs'],
                outputs=example['outputs'],
                **aspect_config
            )
            response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            result = response['message']['content']
        else:
            prompt = template.render(
                inputs=example['inputs'],
                outputs=example['outputs'],
                aspect_config=aspect_config
            )

        output_path = Path(output_dir) / f'{example["id"]}.json'
        eval_output = {
            'inputs': example['inputs'],
            'outputs': example['outputs'],
            'result': result
        }
        
        # the pregen might also contain a previous modification result which should be in the name of the pregen file
        # however if it is not present, it means that there was no previous modification and we just modify the original result
        result_to_modify = example.get('result_premodified', result)

        if eval_mod:
            modified_result = modify_result(prompt, result_to_modify, model, eval_mod, mod_force)
            if modified_result:
                eval_output['result_modified'] = modified_result
            else:
                eval_output['result_modified'] = result # No modification was needed
                logger.info(f"Example {example['id']} had no modification needed, continuing with original result.")

        # logger.info(example['id'] + '\n' + result + '\n\n')
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
    parser.add_argument('--eval-mod', type=str, default=None, help='Modification to apply to the evaluation process')
    parser.add_argument('--mod-force', type=int, help='Force of the severity modification, negative for less severity, positive for more severity')
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
    
    try:
        run_evaluation(
            template,
            aspect_config,
            data,
            args.model,
            args.output_dir,
            args.skip,
            args.limit,
            args.eval_mod,
            args.mod_force
        )
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")

    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")
