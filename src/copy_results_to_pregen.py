from scipy.stats import spearmanr
import os
import json
import argparse
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument("--results-path", "-r", required=True)
parser.add_argument("--pregen-dest-path", "-p", required=True)
parser.add_argument("--model", "-m", required=True, type=str, help='Ollama model name')
args = parser.parse_args()

logger.info(f"Loading eval: {args.results_path}")
fnames = os.listdir(args.results_path)

summary_fname = "scores_summary.json"
if summary_fname in fnames: fnames.remove(summary_fname)


pregen_fname = f"pregen_{args.model}.json"

pregen = []
for fname in fnames:
    
    try:
        with open(f"{args.results_path}/{fname}", 'r') as f:
            data = json.load(f)

        pregen.append(
            {
                'id': fname.replace('.json', ''),
                'inputs': data['inputs'],
                'outputs': data['outputs'],
                'result': data['result'],
                'result_premodified': data['result_modified'],
            }
        )
    except Exception as e:
        logger.error(f"Could not load results from {fname}: {e}")
        continue
        
try:
    with open(f"{args.pregen_dest_path}/{pregen_fname}", 'w') as f:
        json.dump(pregen, f, indent=2)
    print(f"Saved pre-generated results to {args.pregen_dest_path}/{pregen_fname}")
except Exception as e:
    print(f"Error saving pre-generated results: {e}")
    raise e
        
    

