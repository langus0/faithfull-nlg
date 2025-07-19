# MODEL=eval_nemo
# MODEL=eval_gemma
# MODEL=eval_qwen
models=(eval_nemo eval_gemma eval_qwen)

# DATASET=qags
DATASET=hanna
# DATASET=summeval

# ASPECT=factual_consistency
ASPECT=coherence
# ASPECT=relevance
# ASPECT=complexity
# ASPECT=relevance

RESULTS_DIR=results2/eval_mod_results/${DATASET}/${ASPECT}

cascade_types=(0 -1 1)

severity_modification_types=(add_critical_error)
# severity_modification_forces=()

for MODEL in "${models[@]}"
do
    for cascade_type in "${cascade_types[@]}"
    do
        # for sev_force in "${severity_modification_forces[@]}"
        # do
        printf "\nMetrics for modification ${mod_type}${sev_force}\n"
        uv run python src/calculate_error_deletion.py \
            --results-dir ${RESULTS_DIR}/${MODEL}_delete${cascade_type}
            
        # done
    done
done