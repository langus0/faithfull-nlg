import json
import os
import argparse
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from mods_vllm import strip_forbidden_symbols
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", "-r", required=True)
parser.add_argument("--exclude-correlation", action='store_true', help="Calculate Spearman correlation")
parser.add_argument("--exclude-confusion-matrix", action='store_true', help="Calculate confusion matrix")
args = parser.parse_args()

logger.info(f"Loading eval: {args.results_dir}")
fnames = os.listdir(args.results_dir)
summary_fname = "scores_summary.json"


if summary_fname in fnames:
	fnames.remove(summary_fname)

scores = {}
for fname in fnames:
	try:
		with open(f"{args.results_dir}/{fname}", 'r') as f:
			data = json.load(f)
		res, res_mod = data['result'], data['result_modified']
		# print(f"{fname} loaded") # TODO
	except Exception as e:
		logger.warning(f"Couldnt load results from {fname}, skipping")
		continue

	sc, sc_mod = None, None
	for line in res.split('\n'):
		if line.startswith('Overall score'):
			sc = line.split(':')[1].strip()
			sc = strip_forbidden_symbols(sc)
	# if sc i
	for line in res_mod.split('\n'):
		if line.startswith('Overall score'):
			sc_mod = line.split(':')[1].strip()
			sc_mod = strip_forbidden_symbols(sc_mod)

	# if sc is None or sc_mod is None:
	# 	logger.debug(f"A score is None, probably from a N/A evaluation: {sc} {sc_mod}")

	if sc or sc_mod:
		scores[fname] = {
			"score": sc,
			"score_mod": sc_mod
		}


is_modified_map = [
	sc_pair["score"].lower() != sc_pair["score_mod"].lower()
	for sc_pair in scores.values()
	]

mod_sum = sum(is_modified_map)
scores_sum = len(scores)
changed_percent = round(mod_sum / scores_sum * 100, 2)

logger.info(f"Modified {mod_sum} in {scores_sum} examples ({changed_percent}%)")

df = pd.DataFrame(scores).T

mapping = {
    "Unacceptable": 1,
    "Poor": 2,
    "Fair": 3,
    "Good": 4,
    "Excellent": 5
}

# Convert scores to numerical values
df["score"] = df["score"].str.strip()
df["score_mod"] = df["score_mod"].str.strip()

df["score_num"] = df["score"].map(mapping)
df["score_mod_num"] = df["score_mod"].map(mapping)


if not args.exclude_correlation:
	correlation, p_value = spearmanr(df["score_num"], df["score_mod_num"])
	correlation = round(correlation, 4)
	logger.info(f"Spearman correlation: {correlation}")
else:
    correlation = None

if not args.exclude_confusion_matrix:
	matrix = confusion_matrix(df["score_num"], df["score_mod_num"], labels=list(mapping.values()))
	logger.info(f"Confusion matrix: \n{matrix}")
	matrix = matrix.tolist()
else:
    matrix = None

scores_summary = {
    "summary": {
        "analyzed_examples": scores_sum,
		"scores_changed": mod_sum,
		"changed_percent": changed_percent,
		"correlation": correlation,
		"confusion_matrix": matrix
   },
    "scores": scores,
}

with open(f"{args.results_dir}/{summary_fname}", "w") as f:
	json.dump(scores_summary, f, indent=2)