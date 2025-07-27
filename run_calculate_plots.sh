# MODEL=eval_nemo
# MODEL=eval_gemma
# MODEL=eval_qwen
# MODEL=eval_mistral

models=(eval_nemo eval_gemma eval_qwen)

# DATASET=qags
DATASET=hanna
# DATASET=summeval

# ASPECT=factual_consistency
# ASPECT=coherence
# ASPECT=relevance
# ASPECT=complexity

# aspects=(factual_consistency)
aspects=(coherence complexity relevance)
# aspects=(coherence factual_consistency relevance)





for ASPECT in "${aspects[@]}"; do
    for MODEL in "${models[@]}"; do
        RESULTS_DIR=results2/eval_mod_results/${DATASET}/${ASPECT}
        printf "\n\n\nRunning modifications for model ${MODEL} on  ${DATASET}-${ASPECT}\n"
# for mod_type in "${severity_modification_types[@]}"
# do
#     for sev_force in "${severity_modification_forces[@]}"
#     do
        printf "\nMetrics for modification ${mod_type}${sev_force}\n"
        uv run python src/calculate_plots2.py \
            --results-dir ${RESULTS_DIR}/${MODEL}
    done
done

