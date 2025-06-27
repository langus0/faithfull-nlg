import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results-path", "-r", required=True)
args = parser.parse_args()

print(f"Loading eval: {args.results_path}")
fnames = os.listdir(args.results_path)

summary_fname = "scores_summary.json"
if summary_fname in fnames: fnames.remove(summary_fname)

scores = {"scores": {}, "summary": {}}
for fname in fnames:
	try:
		with open(f"{args.results_path}/{fname}", 'r') as f:
			data = json.load(f)
		res, res_mod = data['result'], data['result_modified']
		# print(f"{fname} loaded") # TODO
	except:
		print(f"Couldnt load results from {fname}, skipping")
		continue

	sc, sc_mod = None, None
	for l in res.split('\n'):
		if l.startswith('Overall score'):
			sc = l.split(':')[1].strip()
	for l in res_mod.split('\n'):
		if l.startswith('Overall score'):
			sc_mod = l.split(':')[1].strip()

	if sc is None or sc_mod is None:
		print(f"Some are None! {sc} {sc_mod}")

	if sc or sc_mod:
		scores["scores"][fname] = {
			"score": sc,
			"score_mod": sc_mod
		}


with open(f"{args.results_path}/scores_summary.json", "w") as f:
    json.dump(scores, f)

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

with open(f"{args.results_path}/{summary_fname}", "w") as f:
    json.dump(scores, f)

print(f"Modified {mod_sum} in {scores_sum} examples")
