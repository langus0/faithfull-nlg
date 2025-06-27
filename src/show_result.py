import sys
import argparse
import json
from loguru import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file containing evaluation results')
    args = parser.parse_args()

    json_file = args.json_file
    if not json_file.endswith('.json'):
        logger.error("The provided file is not a JSON file.")
        sys.exit(1)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File {json_file} not found.")
        sys.exit(1)
    
    result = data.get('result', None)
    if result:
        logger.info(f"Evaluation Result:\n{result}\n\n")
    else:
        logger.error("Result not found")

    mod_result = data.get('result_modified', None)
    if mod_result:
        logger.info(f"Modified Result:\n{mod_result}\n\n")
    else:
        logger.error("Modified result not found")
