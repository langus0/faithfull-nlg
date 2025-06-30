import json
import os
import argparse
import pandas as pd
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--results-path", "-r",
                    required=True)
# TODO set below argument as flag
parser.add_argument("--use-scores-summary", "-r",
                    default=False, required=False)
args = parser.parse_args()

print(f"Loading eval: {args.results_path}")
fnames = os.listdir(args.results_path)
summary_fname = "scores_summary.json"

if args.use_scores_summary and summary_fname in fnames:
	summary_path = f"{args.results_path}/scores_summary.json"
	with open(summary_path, "r") as f:
		data = json.load(f)
else:
	if summary_fname in fnames:
		fnames.remove(summary_fname)

	scores = {"scores": {}, "summary": {}}
	for fname in fnames:
		try:
			with open(f"{args.results_path}/{fname}", 'r') as f:
				data = json.load(f)
			res, res_mod = data['result'], data['result_modified']
			# print(f"{fname} loaded") # TODO
		except Exception as e:
			print(f"Couldnt load results from {fname}, skipping")
			continue

		sc, sc_mod = None, None
		for line in res.split('\n'):
			if line.startswith('Overall score'):
				sc = line.split(':')[1].strip()
		for line in res_mod.split('\n'):
			if line.startswith('Overall score'):
				sc_mod = line.split(':')[1].strip()

		if sc is None or sc_mod is None:
			print(f"Some are None! {sc} {sc_mod}")

		if sc or sc_mod:
			scores["scores"][fname] = {
				"score": sc,
				"score_mod": sc_mod
			}

is_modified_map = [
	sc_pair["score"].lower() != sc_pair["score_mod"].lower()
	for sc_pair in scores["scores"].values()
	]

mod_sum = sum(is_modified_map)
scores_sum = len(scores["scores"])

scores["summary"] = {
	"analyzed_examples": len(scores),
	"scores_changed": mod_sum
	}

if not args.use_scores_summary:
	with open(f"{args.results_path}/{summary_fname}", "w") as f:
		json.dump(scores, f)

print(f"Modified {mod_sum} in {scores_sum} examples")

df = pd.DataFrame(data["scores"]).T

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


correlation, p_value = spearmanr(df["score_num"], df["score_mod_num"])
print("Spearman correlation:", correlation)
print("P-value:", p_value)
