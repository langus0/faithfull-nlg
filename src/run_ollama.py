import argparse
import json
import os
from pathlib import Path
from jinja2 import Template

from ollama import chat
from loguru import logger


def run_ollama(
        prompt: str,
        model: str
    ) -> str:
    response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']


def run_evaluation(
        template: Template,
        aspect_config: dict,
        data: dict, model: str,
        output_dir: str
    ):
    for example in data[:10]:
        prompt = template.render(
            inputs=example['inputs'],
            outputs=example['outputs'],
            **aspect_config
        )
        result = run_ollama(prompt, model)
        output_path = Path(output_dir) / f'{example["id"]}.json'
        eval_output = {
            'inputs': example['inputs'],
            'outputs': example['outputs'],
            'result': result
        }
        logger.info(example['id'] + '\n' + result + '\n\n')
        logger.info(f"\nSaving {example['id']} to {output_path}\n\n")
        with open(output_path, 'w') as f:
            f.write(json.dumps(eval_output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ollama with a given prompt and model.")
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to send to the Ollama model.')
    parser.add_argument('--model', type=str, default='eval_nemo', help='The model to use for Ollama.')
    args = parser.parse_args()

    # Run the Ollama chat
    response = run_ollama(args.prompt, args.model)
    logger.info(f"Response from model {args.model}: {response}")
