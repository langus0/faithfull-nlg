import json
import os
import argparse
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from mods import strip_forbidden_symbols
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
   
   
def aggregate_results(results):
    agg = defaultdict(list)
    for score_dict in results.values():
        for num_errors, bool_list in score_dict.items():
            agg[num_errors].extend(bool_list)
    return agg 
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", "-r", required=True)
parser.add_argument("--show_plots", action='store_true')
args = parser.parse_args()
 
MODS = ["delete"]

SEVFORCE = [-1, 1]

mapping = {
    "Unacceptable": 1,
    "Poor": 2,
    "Fair": 3,
    "Good": 4,
    "Excellent": 5
}

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
                res = data['error_mod_impacts']
                # print(f"{fname} loaded") # TODO
            except Exception as e:
                logger.warning(f"Couldnt load results from {fname}, skipping")
                continue
            if not res or len(res) < 1:
                continue
            sc_mod = [errors["new_overall_score"] for errors in res]
            sc_mod = [s.strip() for s in sc_mod]
            sc_mod = [strip_forbidden_symbols(s) for s in sc_mod]
            sc_mod = [mapping[s] for s in sc_mod if s in mapping]  # Filter out empty strings
            sc = res[0]["overall_score"].strip()
            sc = strip_forbidden_symbols(sc)
            sc = mapping[sc]
           

            if fname in scores:
                scores[fname][f"score_{mod_type}{sev_force}"] = sc_mod
            else:
                scores[fname] = {
                        "score": sc, 
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
import numpy as np
from collections import defaultdict
for mod_type in MODS:
    for sev_force in SEVFORCE:
        results = {}
        results2 = {}
        for val in scores.values():
            if f"score_{mod_type}{sev_force}" not in val:
                continue
            n_err = len(val[f"score_{mod_type}{sev_force}"]) +1
            # if val["score"] not in results:
            #     results[val["score"]] = defaultdict(list)
            #     results2[val["score"]] = defaultdict(list)
            if n_err not in results:
                results[n_err] = defaultdict(list)
                results2[n_err] = defaultdict(list)
            r_dict = results[n_err]
            r_dict2 = results2[n_err]
            n_err = n_err -1
            
            for i, key in enumerate(val[f"score_{mod_type}{sev_force}"]):
                if sev_force == 1:
                    r_dict[n_err - i].append(key != val["score"])
                    r_dict2[n_err - i].append(key - val["score"])
                else:
                    r_dict[i + 1].append(key != val["score"])
                    r_dict2[i + 1].append(key - val["score"])
                    
        #results has structire {ovreall_score: {num_errors: [bool, ...]}}
        #create a plot with line showing percentage of changes (bool, y axis) for each num_errors (x axis). A line should be ploted for each oveall_score.
        
        for r,title in zip([results, results2], ["Proportion of Changed Predictions", "Average Score Change"]):
            plt.figure(figsize=(10, 6))

            for overall_score in range(2, 6):
                error_dict = r.get(overall_score, {})
                x = sorted(error_dict.keys())
                #add 0 at the beginning of x
                
                y_vals = [error_dict[k] for k in x] 

                # Compute mean and std for each x
                means = [np.mean(vals) for vals in y_vals]
                stds = [np.std(vals) / np.sqrt(len(vals)) for vals in y_vals]
                x = [0] + x
                means = [0] + means
                stds = [0] + stds

                # Plot mean line
                plt.plot(x, means, label=f"{overall_score} error(s)")

                # Plot standard deviation band
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
                plt.fill_between(x, lower, upper, alpha=0.2)

            plt.xlabel("Number of Deleted Errors")
            plt.ylabel(title)
            #set x tricks to be 0, 1, 2, 3, 4, 5
            plt.xticks(range(0, 5))
            if title == "Proportion of Changed Predictions":
                plt.ylim(0, 1)
            else:
                plt.ylim(-0.2, 1)
            #plt.title("Score Change Proportion vs. Number of Errors")
            plt.legend(title="Numer of Example's Errors")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{args.results_dir}n_err_{mod_type}{sev_force}_{title}.png", dpi=300, bbox_inches='tight')
            if args.show_plots:
                plt.show()
        
        
        
        # Aggregate both results
        agg1 = aggregate_results(results)
        agg2 = aggregate_results(results2)

        # Get shared sorted x values
        x_all = sorted(set(agg1.keys()) | set(agg2.keys()))
        # insert 0 at the beginning of x_all
        x_all = [0] + x_all
        # Fill in missing keys with empty lists to avoid KeyError
        for x in x_all:
            agg1.setdefault(x, [])
            agg2.setdefault(x, [])

        # Compute means and stds
        means1 = [np.mean(agg1[k]) if agg1[k] else 0 for k in x_all]
        stds1 = [np.std(agg1[k]) / np.sqrt(len(agg1[k])) if agg1[k] else 0 for k in x_all]

        means2 = [np.mean(agg2[k]) if agg2[k] else 0 for k in x_all]
        stds2 = [np.std(agg2[k]) / np.sqrt(len(agg2[k])) if agg2[k] else 0 for k in x_all]
        #instert 0 at the beginning of means and stds
        # means1 =  means1
        # stds1 = stds1
        # means2 =  means2
        # stds2 =  stds2
        logger.info(f"X: {x_all}")
        logger.info(f"Means: {means1}")
        logger.info(f"Avg for {x_all[1]}: {means1[1]}")
        logger.info(f"avg for {x_all[2]}: {means1[2]}")
        # Create plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # First y-axis (left)
        ax1.plot(x_all, means1, color="blue", label="Proportion of changes")
        ax1.fill_between(x_all, [m - s for m, s in zip(means1, stds1)],
                                [m + s for m, s in zip(means1, stds1)],
                        color="blue", alpha=0.2)
        ax1.set_xlabel("Number of Deleted Errors")
        ax1.set_ylabel("Proportion of changes", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_ylim(0,1)
        ax1.set_xlim(0,5)

        # Second y-axis (right)
        ax2 = ax1.twinx()
        ax2.plot(x_all, means2, color="red", label="Avg Score Change")
        
        ax2.fill_between(x_all, [m - s for m, s in zip(means2, stds2)],
                                [m + s for m, s in zip(means2, stds2)],
                        color="red", alpha=0.2)
        ax2.set_ylabel("Avg Score Change", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Title and grid
        plt.title("Comparison of Score Change Proportions (results vs results2)")
        fig.tight_layout()
        plt.savefig(f"{args.results_dir}n_err2_{mod_type}{sev_force}.png", dpi=300, bbox_inches='tight')
        plt.grid(True)
        if args.show_plots:
            plt.show()
exit()
      

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
if args.show_plots:
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
if args.show_plots:
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
if args.show_plots:
    plt.show()
exit()
