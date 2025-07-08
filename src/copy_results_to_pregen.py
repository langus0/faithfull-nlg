from scipy.stats import spearmanr
import os
import json
import argparse
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", "-r", required=True)
parser.add_argument("--pregen-dest-dir", "-p", required=True)
parser.add_argument("--pregen-tag", "-n", required=True, type=str, help='tag of the pregen file')
parser.add_argument("--exclude-premodified-result", "-e", action='store_true', help='Exclude pre-modified result from the output')
args = parser.parse_args()

logger.info(f"Loading eval: {args.results_dir}")
fnames = os.listdir(args.results_dir)

summary_fname = "scores_summary.json"
if summary_fname in fnames: fnames.remove(summary_fname)


pregen_fname = f"pregen_{args.pregen_tag}.json"

pregen = []
for fname in fnames:
    
    try:
        with open(f"{args.results_dir}/{fname}", 'r') as f:
            data = json.load(f)

        example = {
                'id': fname.replace('.json', ''),
                'inputs': data['inputs'],
                'outputs': data['outputs'],
                'result': data['result'],
            }
        if not args.exclude_premodified_result:
            example['result_premodified'] = data['result_modified']
        pregen.append(example)
    except Exception as e:
        logger.error(f"Could not load results from {fname}: {e}")
        continue

logger.info(f"Saving to pregen: {args.pregen_dest_dir}/{pregen_fname}")

try:
    with open(f"{args.pregen_dest_dir}/{pregen_fname}", 'w') as f:
        json.dump(pregen, f, indent=2)
    print(f"Saved pre-generated results to {args.pregen_dest_dir}/{pregen_fname}")
except Exception as e:
    print(f"Error saving pre-generated results: {e}")
    raise e
        
    

