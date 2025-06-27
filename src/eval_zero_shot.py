import argparse
import json
import os
import time
from pathlib import Path
from jinja2 import Template
from typing import Callable

from ollama import chat
from loguru import logger

def modify_severity_parser(
        result: str
    ) -> str:
    """
    Parse the result from the model and increase the severity of the error.
    Returns the modified result with increased severity.
    If no Overall score is found, it continues with generation.
    If "No Error" is found, it returns None to signal that no modification is needed.
    """
    
    modified_result = []
    
    lines = result.strip().split('\n')
    for line in lines:
        if line.startswith("Severity:"):
            try:
                severity_parts = line.split(':')[1].split()
                severity = int(severity_parts[0].strip())
                new_line = f"Severity: {max(0, severity - 2)}"
                if len(severity_parts) > 2:
                    new_line += ' ' + ' '.join(severity_parts[1:])
                modified_result.append(new_line)
            except:
                logger.warning(f"Failed to parse severity from line: {line}")
                modified_result.append(line)
        elif line.startswith("Overall score:"):
            modified_result.append("Overall score:")
            return '\n'.join(modified_result)
        elif line.startswith("No Error"):
            return None # No modification or further generation needed
        else:
            modified_result.append(line)
    
    logger.warning("No Overall score found, continuing with generation.")
    return '\n'.join(modified_result)

EVAL_MODS = {
    "severity": modify_severity_parser,
}

def modify_result(
        prompt: str,
        result: str,
        eval_mod: str,
        model: str
    ) -> str:
    """
    Parse the result from the model based on the evaluation modification.
    This function can use different functions for error modification.
    """
    
    modified_result = EVAL_MODS[eval_mod](result)

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
        eval_mod: str
    ):

    data = data[skip:limit + skip if limit else None]
    
    for i, example in enumerate(data):
        prompt = template.render(
            inputs=example['inputs'],
            outputs=example['outputs'],
            **aspect_config
        )
        response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        result = response['message']['content']

        output_path = Path(output_dir) / f'{example["id"]}.json'
        eval_output = {
            'inputs': example['inputs'],
            'outputs': example['outputs'],
            'result': result
        }
        
        if eval_mod:
            modified_result = modify_result(prompt, result, eval_mod, model)
            if modified_result:
                eval_output['result_modified'] = modified_result
            else:
                eval_output['result_modified'] = result # No modification was needed

        # logger.info(example['id'] + '\n' + result + '\n\n')
        logger.info(f"\n[{i}] Saving {example['id']} to {output_path}\n")
        with open(output_path, 'w') as f:
            f.write(json.dumps(eval_output, indent=2))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, help='Path to the prompt template file')
    parser.add_argument('--aspect-config', type=str, help='Path to the aspect configuration file')
    parser.add_argument('--data', type=str, help='Path to the datasets with inputs and outputs to evaluate')
    parser.add_argument('--model', type=str, default='eval_nemo', help='Ollama model name')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--skip', type=int, default=0, help='Slice of examples to evaluate if there should be less than all')
    parser.add_argument('--limit', type=int, default=None, help='Slice of examples to evaluate if there should be less than all')
    parser.add_argument('--eval-mod', type=str, default=None, help='Modification to apply to the evaluation process')
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
            args.eval_mod
        )
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")

    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")
