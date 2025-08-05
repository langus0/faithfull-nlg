### STEP 1: Choose model

# MODEL=eval_nemo
MODEL=eval_gemma
# MODEL=eval_qwen


### STEP 2: Choose dataset & set of aspects

# DATASET=qags
# aspects=(factual_consistency)

DATASET=hanna
aspects=(coherence complexity relevance)

# DATASET=summeval
# aspects=(coherence factual_consistency relevance)


### STEP 3: Modification-specific parameters


# severity_modification_types=(severity textsev int_and_textsev)
# severity_modification_forces=(1 2 -1 -2)

# severity_modification_types=(add_critical_error)
severity_modification_types=(add_random_error2)


for ASPECT in "${aspects[@]}"
do
    RESULTS_DIR=results/eval_mod_results/${DATASET}/${ASPECT}
    for mod_type in "${severity_modification_types[@]}"
    do
        printf "\n\n\nMetrics for ${MODEL} modification ${mod_type}\n\n"
        uv run python src/calculate_metric.py \
            --results-dir ${RESULTS_DIR}/${MODEL}_${mod_type} \
            --exclude-confusion-matrix \
            --exclude-correlation
    done
done