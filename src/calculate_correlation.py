from scipy.stats import spearmanr
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results-path", "-r", required=True)
args = parser.parse_args()

summary_path = f"{args.results_path}/scores_summary.json"
with open(summary_path, "r") as f:
    data = json.load(f)
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