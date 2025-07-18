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



severity_modification_types=(severity textsev int_and_textsev)

severity_modification_forces=(1 2 -1 -2)

for MODEL in eval_nemo eval_gemma; do
    for ASPECT in relevance coherence factual_consistency; do
        RESULTS_DIR=results3/eval_mod_results/${DATASET}/${ASPECT}
        printf "\nRunning modifications for model ${MODEL} on aspect ${ASPECT}\n"
# for mod_type in "${severity_modification_types[@]}"
# do
#     for sev_force in "${severity_modification_forces[@]}"
#     do
        printf "\nMetrics for modification ${mod_type}${sev_force}\n"
        python src/calculate_plots.py \
            --results-dir ${RESULTS_DIR}/${MODEL}
    done
done

