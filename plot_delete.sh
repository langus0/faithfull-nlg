### STEP 1: Choose model

# All models
models=(eval_nemo eval_gemma eval_qwen eval_mistral)


### STEP 2: Choose dataset & set of aspects

# DATASET=qags
# aspects=(factual_consistency)

DATASET=hanna
aspects=(coherence complexity relevance)

# DATASET=summeval
# aspects=(coherence factual_consistency relevance)


for ASPECT in "${aspects[@]}"; do
    for MODEL in "${models[@]}"; do
        RESULTS_DIR=results/eval_mod_results/${DATASET}/${ASPECT}
        printf "\n\n\nPlotting results of model ${MODEL} on ${DATASET}-${ASPECT}\n"
        uv run python src/calculate_plots2.py \
            --results-dir ${RESULTS_DIR}/${MODEL}
    done
done

