### STEP 1: Choose model

# MODEL=eval_nemo
MODEL=eval_gemma
# MODEL=eval_qwen
# MODEL=eval_mistral


### STEP 2: Choose dataset & set of aspects

# DATASET=qags
# aspects=(factual_consistency)

DATASET=hanna
aspects=(coherence complexity relevance)

# DATASET=summeval
# aspects=(coherence factual_consistency relevance)


### STEP 3: Modification-specific parameters

cascade_types=(0 -1 1)


for ASPECT in "${aspects[@]}"
do
    RESULTS_DIR=results/eval_mod_results/${DATASET}/${ASPECT}

    for cascade_type in "${cascade_types[@]}"
    do
        printf "\nMetrics for modification ${mod_type}${sev_force}\n"
        uv run python src/calculate_error_deletion.py \
            --results-dir ${RESULTS_DIR}/${MODEL}_delete${cascade_type}
            
        # done
    done
done