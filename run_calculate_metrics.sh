MODEL=eval_nemo
# MODEL=eval_gemma
# MODEL=eval_qwen
# MODEL=eval_mistral

# DATASET=qags
# DATASET=hanna
DATASET=summeval

ASPECT=factual_consistency
# ASPECT=coherence
# ASPECT=relevance
# ASPECT=complexity

RESULTS_DIR=results/eval_mod_results/${DATASET}/${ASPECT}

severity_modification_types=(severity textsev int_and_textsev)

severity_modification_forces=(1 2 -1 -2)



for mod_type in "${severity_modification_types[@]}"
do
    for sev_force in "${severity_modification_forces[@]}"
    do
        printf "\nMetrics for modification ${mod_type}${sev_force}\n"
        uv run python src/calculate_metric.py \
            --results-dir ${RESULTS_DIR}/${MODEL}_${mod_type}${sev_force}
    done
done