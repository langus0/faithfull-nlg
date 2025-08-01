import json
import os
import argparse
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from mods import strip_forbidden_symbols
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", "-r", required=True)
args = parser.parse_args()

logger.info(f"Loading eval: {args.results_dir}")
fnames = os.listdir(args.results_dir)
summary_fname = "scores_summary.json"

if summary_fname in fnames:
	fnames.remove(summary_fname)

impacts_groups = []
for fname in fnames:
	try:
		with open(f"{args.results_dir}/{fname}", 'r') as f:
			data = json.load(f)
		impacts_groups.append(data['error_mod_impacts'])
	except Exception as e:
		logger.warning(f"Couldnt load results from {fname}, skipping")
		continue

cascade_direction = int(args.results_dir.split("delete")[-1])

indices_map = {
	-1: {1: (0, 1), 2: (1, 2)},
	0: {1: (None, None)},
	1: {1: (-1, None), 2: (-2, -1)},
}

indices = indices_map[cascade_direction]

impacts_flat = {k: [] for k in indices.keys()}

for n_error_deletions, deletion_indices in impacts_flat.keys():
	for impacts in impacts_groups:
		impact_list = impacts[deletion_indices[0]:deletion_indices[1]]

		impacts_flat[n_error_deletions].extend(impact_list)

for n_error_deletions, impacts in impacts_flat.items():
	scores = [{
		"score": strip_forbidden_symbols(impact["overall_score"]),
		"new_score": strip_forbidden_symbols(impact["new_overall_score"])
		}
		for impact in impacts
	]
	is_modified_map = [
		sc_pair["score"].lower() != sc_pair["new_score"].lower()
		for sc_pair in scores
	]

	mod_sum = sum(is_modified_map)
	scores_sum = len(impacts)
	changed_percent = round(mod_sum / scores_sum * 100, 2)

	logger.info(f"Modified {mod_sum} in {scores_sum} examples ({changed_percent}%)")

	scores_summary = {
			"summary": {
				"analyzed_examples": scores_sum,
				"scores_changed": mod_sum,
				"changed_percent": changed_percent,
		},
		"scores": scores,
	}

	with open(f"{args.results_dir}/{summary_fname}-{n_error_deletions}.json", "w") as f:
		json.dump(scores_summary, f, indent=2)
