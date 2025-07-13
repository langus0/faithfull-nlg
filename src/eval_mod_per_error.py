import argparse
import json
import os
import time
from pathlib import Path
from jinja2 import Template
from typing import Callable

from ollama import chat
from loguru import logger

from mods import modify_impact_per_error, modify_delete_per_error

EVAL_MODS = {
    "impact": modify_impact_per_error,
    "delete": modify_delete_per_error,
    "none": lambda x, y, z: None
}

def run_evaluation(
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
    
    for i, example in enumerate(data):
        
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
            response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            result = response['message']['content']

        output_path = Path(output_dir) / f'{example["id"]}.json'
        eval_output = {
            'result': result,
            'error_mod_impacts': []
        }

        error_mod_impacts = EVAL_MODS[per_error_mod](prompt, result, model, mod_direction)
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
    
    try:
        run_evaluation(
            template,
            aspect_config,
            data,
            args.model,
            args.output_dir,
            args.skip,
            args.limit,
            args.per_error_mod,
            args.mod_direction,
        )
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        # raise e

    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")
