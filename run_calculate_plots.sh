#MODEL=eval_nemo
MODEL=eval_gemma
# MODEL=eval_qwen
# MODEL=eval_mistral

# DATASET=qags
# DATASET=hanna
DATASET=summeval

ASPECT=factual_consistency
# ASPECT=coherence
# ASPECT=relevance
# ASPECT=complexity





for MODEL in eval_nemo eval_gemma eval_qwen; do
    for ASPECT in factual_consistency relevance coherence; do
        RESULTS_DIR=results2/eval_mod_results/${DATASET}/${ASPECT}
        printf "\nRunning modifications for model ${MODEL} on aspect ${ASPECT}\n"
# for mod_type in "${severity_modification_types[@]}"
# do
#     for sev_force in "${severity_modification_forces[@]}"
#     do
        printf "\nMetrics for modification ${mod_type}${sev_force}\n"
        python src/calculate_plots2.py \
            --results-dir ${RESULTS_DIR}/${MODEL}
    done
done

