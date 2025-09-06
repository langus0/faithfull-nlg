import json
import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from mods import strip_forbidden_symbols
from loguru import logger
import matplotlib.pyplot as plt
from textwrap import fill
from matplotlib.ticker import PercentFormatter

# Assuming df is your DataFrame
def classify_row(row, mods):
    main = row['score_num']
    add_rand1 = row[f'score_num_{mods[0]}']
    add_rand2 = row[f'score_num_{mods[1]}']
    add_crit = row[f'score_num_{mods[2]}']


    if add_crit < add_rand2 < add_rand1 < main:
        return 'lower_cons'
    elif add_crit < add_rand2 < add_rand1 == main:
        return 'lower_after_rand2'
    elif add_crit < add_rand2 == add_rand1 == main:
        return 'lower_after_crit'
    elif  add_rand2 < add_rand1 < main == add_crit:
        return 'lower_only_rand1'
    elif  add_rand2 < add_rand1 == main == add_crit:
        return 'lower_only_rand2'
    if add_crit == add_rand2 == add_rand1 == main:
        return 'equal_all'
    elif np.any(np.array([add_crit, add_rand1, add_rand2]) > main):
        return 'incr'

    return 'rest'
    
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", "-r", required=True)
parser.add_argument("--show_plots", action='store_true')
args = parser.parse_args()

MODS = ["add_random_error1", "add_random_error2", "add_critical_error"]


scores = {}
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
for mod_type in MODS:
    res_dir = f"{args.results_dir}_{mod_type}"

    logger.info(f"Loading eval: {res_dir}")
    fnames = os.listdir(res_dir)
    summary_fname = "scores_summary.json"


    if summary_fname in fnames:
        fnames.remove(summary_fname)

    
    for fname in fnames:
        try:
            with open(f"{res_dir}/{fname}", 'r') as f:
                data = json.load(f)
            res, res_mod = data['result'], data['result_modified']
            # print(f"{fname} loaded") # TODO
        except Exception as e:
            logger.warning(f"Couldnt load results from {fname}, skipping")
            continue

        sc, sc_mod = None, None
        severities= []
        for line in res.split('\n'):
            if line.startswith('Overall score'):
                if sc is not None:
                    logger.warning(f"Duplicate Overalscore in result {fname}")
                sc = line.split(':')[1].strip()
                sc = strip_forbidden_symbols(sc)
            if line.startswith('Severity:'):
                try:
                    severity_parts = line.split(':')[1].split()
                    severity = int(severity_parts[0].strip())
                    severities.append(severity)
                except Exception as e:
                    #logger.error(f"Error parsing severity from {fname}: {line}")
                    severities.append(0)
        # if sc i
        for line in res_mod.split('\n'):
            if line.startswith('Overall score'):
                if sc_mod is not None:
                    logger.warning(f"Duplicate Overalscore in mod {fname}")
                sc_mod = line.split(':')[1].strip()
                sc_mod = strip_forbidden_symbols(sc_mod)

        # if sc is None or sc_mod is None:
        # 	logger.debug(f"A score is None, probably from a N/A evaluation: {sc} {sc_mod}")

        if sc or sc_mod:
            if fname in scores:
                scores[fname][f"score_{mod_type}"] = sc_mod
            else:
                scores[fname] = {
                    "score": sc, "severities": severities,
                    f"score_{mod_type}": sc_mod}
                


# is_modified_map = [
# 	sc_pair["score"].lower() != sc_pair["score_mod"].lower()
# 	for sc_pair in scores.values()
# 	]

# mod_sum = sum(is_modified_map)
# scores_sum = len(scores)
# changed_percent = round(mod_sum / scores_sum * 100, 2)

# logger.info(f"Modified {mod_sum} in {scores_sum} examples ({changed_percent}%)")

df = pd.DataFrame(scores).T

mapping = {
    "Unacceptable": 1,
    "Poor": 2,
    "Fair": 3,
    "Good": 4,
    "Excellent": 5
}
print(df.columns)
# Convert scores to numerical values
df["score"] = df["score"].str.strip()
df["score_num"] = df["score"].map(mapping)
for mod_type in MODS:
    df[f"score_{mod_type}"] = df[f"score_{mod_type}"].str.strip()
    df[f"score_num_{mod_type}"] = df[f"score_{mod_type}"].map(mapping)
# df["score_mod"] = df["score_mod"].str.strip()

#add column with lenthg of list in severities column
df["severities_length"] = df["severities"].apply(lambda x: len(x) if isinstance(x, list) else 0)

# df["score_num"] = df["score"].map(mapping)
# df["score_mod_num"] = df["score_mod"].map(mapping)
column = 'score_num'

plt.figure(figsize=(8, 5))
plt.hist(df[column], bins=20, edgecolor='black', color='skyblue')

plt.title(f'Histogram of {column}')
plt.xlabel(column)
plt.ylabel('Frequency')

# Save to file (optional)
plt.savefig(f"{args.results_dir}_error_addition_histogram_{column}.png", dpi=300, bbox_inches='tight')
if args.show_plots:
    plt.show()

print(df.groupby('score_num')['severities_length'].value_counts())


df_all = None
#iterate over rows of df and create new df with (len(severity), score_num-score_num_sev_forece)) columns
for mod_type in MODS:
    df_combined = None
    df_tmp = pd.DataFrame({
        'sev': df['severities_length'],
        'change': df[f'score_num_{mod_type}'] - df['score_num']
    })
    if df_combined is None:
        df_combined = df_tmp
    else:
        df_combined = pd.concat([df_combined, df_tmp], axis=0)
    #Group df_combined by sev and compute mean of "change" columnt

    # df_combined = df_combined.groupby('sev')['change'].mean()
    # df_combined = df_combined.reset_index(name=f'change{mod_type}')
    # df_combined.columns = ['sev', f'change{mod_type}']
    grouped = df_combined.groupby('sev').filter(lambda x: len(x) >= 2)
    grouped = grouped.groupby('sev')['change']
    #print(grouped)
    
    df_combined = grouped.mean().reset_index(name='mean_change')
    df_combined['sem_change'] = grouped.sem().values  # Standard Error of the Mean
    df_combined.columns = ['sev', f'change{mod_type}', f'std{mod_type}']
    print(df_combined.columns)
    if df_all is None:
        df_all = df_combined
    else:
        print(df_combined.columns)
        print(df_all.columns)
        df_all = pd.merge(df_all, df_combined, on='sev', how='outer')
    print(df_all)
    print(df_all.columns)

MODS_NAMES = {"add_critical_error":"Critical error","add_random_error1":"One random error", "add_random_error2":"Two random errors"}
#plot df_all with sev on x axis and change on y axis
plt.figure(figsize=(10, 6))
# plt.plot(df_all['sev'], df_all.filter(like='change'), marker='o', linestyle='-')
for mod_type in MODS:
    # plt.errorbar(
    #     df_all['sev'],
    #     df_all[f'change{mod_type}'],
    #     yerr=df_all[f'std{mod_type}'],
    #     marker='o',
    #     linestyle='-',
    #     capsize=5,
    #     label='Change'
    # )
    plt.plot(df_all['sev'], df_all[f'change{mod_type}'], marker='o', label=f"{MODS_NAMES[mod_type]}")
    plt.fill_between(
    df_all['sev'],
    df_all[f'change{mod_type}'] - df_all[f'std{mod_type}'],
    df_all[f'change{mod_type}'] + df_all[f'std{mod_type}'],
     alpha=0.2
)
#plt.title('Average Change in Score by Severity Modification')
plt.xlabel('Total Severity Modification (sum)')
#add horizontal line at y=0
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('Average Change in Overall Score')
plt.legend()#["Severity Score", "_", "Explanation", "_", "Both"], title='Modification Type')
#plt.xticks(df_all['sev'])
#plt.grid()
plt.savefig(f"{args.results_dir}_error_addition_average_change.png", dpi=300, bbox_inches='tight')
if args.show_plots:
    plt.show()

# logger.info(f"Dataframe:\n{df.columns}")
# print(df.head(5))
# exit()

# fig = plt.figure(figsize=(18, 10))  # Adjust figsize as needed
#axes = axes.flatten()  # So we can index as a flat list
df['category'] = df.apply(lambda x:classify_row(x,MODS), axis=1)
#filter out scores with severities_length ==0
df_filtered = df[df['severities_length'] > 0]
grouped = (
    df_filtered.groupby('score_num')['category']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

nice_labels = {
    'equal_all': 'No change after any modification',
    'lower_cons': 'Consistently lower when adding errors',
    'lower_after_rand2': 'Lowering only after adding 2 errors or critical error',
    'lower_after_crit': 'Consistently only after adding critical error',
    'lower_only_rand1': 'Lowering only after adding one random error, not critical',
    'lower_only_rand2': 'Lowering only after adding two random errors, not critical',
    'incr': 'Inconsistency - score increased after modification',
    'rest': 'Inconsistent behaviour',
}

all_categories = nice_labels.keys()
for cat in all_categories:
    if cat not in grouped.columns:
        grouped[cat] = 0

# Sort columns for consistent color stacking
grouped = grouped[all_categories]

#Apply fill(l, 20) for each nice_label value
nice_labels = {k: fill(v, 22) for k, v in nice_labels.items()}

# Rename columns to nice labels
grouped = grouped.rename(columns=nice_labels)

# Plot
# grouped.plot(kind='bar', stacked=True,  colormap='viridis',  width=0.95, ax=axes[0,i], legend=False )
fig, ax = plt.subplots(figsize=(12, 6))
grouped.plot(kind='bar', stacked=True, colormap='viridis', width=0.95, ax=ax)

ax.set_ylabel('Percentage of Examples')
ax.set_xlabel('Overall Score')
ax.set_title("Distribution of score changes after addition")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xticklabels(grouped.index, rotation=0)

ax.legend(
    title="Category",
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0.,
    frameon=False
)

fig.tight_layout()

plt.savefig(f"{args.results_dir}_error_addition_score_distribution.png")  # or use .pdf, .svg, etc.
if args.show_plots:
    plt.show()
exit()
