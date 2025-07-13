import os
import json
import argparse
from loguru import logger
from ollama import chat

from .templates.perturbations.perturbation_prompts import aspect_perturbation_prompts

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", "-d", required=True)
parser.add_argument("--model", "-m", required=True, type=str, help='Model to use for perturbation')
parser.add_argument("--aspect", "-a", required=True, type=str, help='Aspect under which the inputs should be perturbed')
parser.add_argument('--skip', type=int, default=0, help='Slice of examples to evaluate if there should be less than all')
parser.add_argument('--limit', type=int, default=None, help='Slice of examples to evaluate if there should be less than all')
args = parser.parse_args()

dataset_path = f"data/meta_eval/{args.dataset_name}.json"
destination_path = f"results/pregen_results/perturbed/by_{args.model}/{args.dataset_name}-{args.aspect}.json"

try:
    with open(dataset_path, 'r') as f:
        data = json.load(f)
except Exception as e:
    logger.error(f"Could not load results from {dataset_path}: {e}")
    exit(1)

data = data[args.skip:args.limit + args.skip if args.limit else None]

logger.info(f"Loaded data from {dataset_path} with {len(data)} examples.")

try:
    perturbation_prompt = aspect_perturbation_prompts[args.aspect]
except KeyError:
    logger.error(f"Aspect {args.aspect} not found in prompts. Available aspects: {list(aspect_perturbation_prompts.keys())}")
    exit(1)

logger.info(f"Using perturbation prompt for aspect {args.aspect}")

results = []
for i, example in enumerate(data):
    
    outputs = example.get('outputs', {})
    if not outputs:
        logger.error(f"No outputs found for example {example.get('id', 'unknown')}")
        continue    
    key = list(outputs.keys())[0]
    text = list(outputs.values())[0]
    
    prompt = perturbation_prompt.replace("{{example}}", text)
    
    logger.info(f"\n[{i}] Generating {example['id']}")
    
    response = chat(
                    model=args.model,
                    messages=[
                        {'role': 'user', 'content': prompt},
                    ]
                )
                
    modified_text = response['message']['content'].strip()
    results.append({
        'id': example['id'],
        'inputs': example["inputs"],
        'outputs': {key: modified_text},
    })
    
    logger.info(f"Generated result for {example['id']}")

os.makedirs(os.path.dirname(destination_path), exist_ok=True)
with open(destination_path, 'w') as f:
    json.dump(results, f, indent=2)
