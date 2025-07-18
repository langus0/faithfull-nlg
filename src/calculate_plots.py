import json
import os
import argparse
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from mods_vllm import strip_forbidden_symbols
from loguru import logger
import matplotlib.pyplot as plt
from textwrap import fill
# Assuming df is your DataFrame
def classify_row(row, mod_type, direction=1):
    main = row['score_num']
    pos1 = row[f'score_num_{mod_type}1']
    pos2 = row[f'score_num_{mod_type}2']
    neg1 = row[f'score_num_{mod_type}-1']
    neg2 = row[f'score_num_{mod_type}-2']

    if direction == 1:
        if main == pos1 == pos2:
            return 'equal_all'
        elif pos1 < main and pos2 < main:
            return 'lower_pos1_pos2'
        elif pos2 < main and not (pos1 < main):
            return 'lower_pos2_only'

    else:
        if main == neg1 == neg2:
            return 'equal_all'
        elif neg1 > main and neg2 > main:
            return 'higher_neg1_neg2'
        elif neg2 > main and not (neg1 > main):
            return 'higher_neg2_only'


    return 'rest'
    
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", "-r", required=True)
args = parser.parse_args()
 
MODS = ["severity","textsev", "int_and_textsev"]

SEVFORCE = [-1, 1,-2,2]

scores = {}
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
for mod_type in MODS:
    for sev_force in SEVFORCE:
        res_dir = f"{args.results_dir}_{mod_type}{sev_force}"

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
                    scores[fname][f"score_{mod_type}{sev_force}"] = sc_mod
                else:
                    scores[fname] = {
                        "score": sc, "severities": severities,
                        f"score_{mod_type}{sev_force}": sc_mod}
                


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
    for sev_force in SEVFORCE:
        df[f"score_{mod_type}{sev_force}"] = df[f"score_{mod_type}{sev_force}"].str.strip()
        df[f"score_num_{mod_type}{sev_force}"] = df[f"score_{mod_type}{sev_force}"].map(mapping)
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
plt.savefig(f"{args.results_dir}histogram_{column}.png", dpi=300, bbox_inches='tight')
plt.show()

print(df.groupby('score_num')['severities_length'].value_counts())


df_all = None
#iterate over rows of df and create new df with (len(severity)*sev_force, score_num-score_num_sev_forece)) columns
for mod_type in MODS:
    df_combined = None
    for sev_force in SEVFORCE:
        df_tmp = pd.DataFrame({
            'sev': df['severities_length'] * sev_force,
            'change': df[f'score_num_{mod_type}{sev_force}'] - df['score_num']
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

MODS_NAMES = {"severity":"Severity Score","textsev":"Explanation", "int_and_textsev":"Both"}
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
plt.savefig(f"{args.results_dir}average_change_by_severity.png", dpi=300, bbox_inches='tight')
plt.show()
        
        
DIR_NAME = {1: " (increasing severity)", -1: " (decreasing severity)"}
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust figsize as needed
#axes = axes.flatten()  # So we can index as a flat list
for i, mod_type in enumerate(MODS):
    for j, direction in enumerate([1, -1]):
        df['category'] = df.apply(lambda x:classify_row(x,mod_type, direction=direction), axis=1)
        #filter out scores with severities_length ==0
        df_filtered = df[df['severities_length'] > 0]
        grouped = df_filtered.groupby('score_num')['category'].value_counts(normalize=True).unstack(fill_value=0)
        if direction ==1:
            all_categories = [ 'rest', 'lower_pos2_only', 'lower_pos1_pos2', 'equal_all']
        else:
            all_categories = [ 'rest', 'equal_all', 'higher_neg1_neg2', 'higher_neg2_only',]
        all_categories = [ 'equal_all', 'higher_neg1_neg2', 'higher_neg2_only',  'lower_pos1_pos2','lower_pos2_only', 'rest']
        for cat in all_categories:
            if cat not in grouped.columns:
                grouped[cat] = 0

        # Sort columns for consistent color stacking
        grouped = grouped[all_categories]

        nice_labels = {
            'rest': 'Inconsistent behaviour',
            'lower_pos2_only': 'Lower after increasing severity +2',
            'lower_pos1_pos2': 'Lower after increasing severity +1',
            'equal_all': 'No change after any modification',
            'higher_neg1_neg2': 'Higher after decreasing severity -1',
            'higher_neg2_only': 'Higher after decreasing severity -2',
            
        }
        
        #Apply fill(l, 20) for each nice_label value
        nice_labels = {k: fill(v, 22) for k, v in nice_labels.items()}

        # Rename columns to nice labels
        grouped = grouped.rename(columns=nice_labels)

        # Plot
        grouped.plot(kind='bar', stacked=True,  colormap='viridis',  width=0.95, ax=axes[j,i], legend=False )
        if i == 0:
            axes[j,i].set_ylabel('Percentage of Examples')
        axes[j,i].set_xlabel('Overall Score')
        axes[j,i].set_title(MODS_NAMES[mod_type]+DIR_NAME[direction])
        axes[j,i].set_xticklabels(grouped.index, rotation=0)
    #plt.xticks(rotation=0)
    #plt.title('Score Changes by score_num')
    #plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Category", loc='center right')
    
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.savefig(f"{args.results_dir}score_distribution.png", dpi=300, bbox_inches='tight')  # or use .pdf, .svg, etc.
plt.show()
exit()

correlation, p_value = spearmanr(df["score_num"], df["score_mod_num"])
correlation = round(correlation, 4)
logger.info(f"Spearman correlation: {correlation}")

matrix = confusion_matrix(df["score_num"], df["score_mod_num"], labels=list(mapping.values()))
logger.info(f"Confusion matrix: \n{matrix}")

scores_summary = {
    "summary": {
        "analyzed_examples": scores_sum,
		"scores_changed": mod_sum,
		"changed_percent": changed_percent,
		"correlation": correlation,
		"confusion_matrix": matrix.tolist()
   },
    "scores": scores,
}

with open(f"{args.results_dir}/{summary_fname}", "w") as f:
	json.dump(scores_summary, f, indent=2)