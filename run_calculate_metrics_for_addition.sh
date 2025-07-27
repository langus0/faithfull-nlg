# MODEL=eval_nemo
# MODEL=eval_gemma
# MODEL=eval_qwen
# models=(eval_nemo eval_gemma eval_qwen)
models=(eval_qwen)

# DATASET=qags
# DATASET=hanna
DATASET=summeval

# ASPECT=factual_consistency
# ASPECT=coherence
ASPECT=relevance
# ASPECT=complexity

RESULTS_DIR=results2/eval_mod_results/${DATASET}/${ASPECT}

# severity_modification_types=(severity textsev int_and_textsev)
# severity_modification_forces=(1 2 -1 -2)

# severity_modification_types=(add_critical_error)
severity_modification_types=(add_random_error2)

for MODEL in "${models[@]}"
do
    for mod_type in "${severity_modification_types[@]}"
    do
        printf "\n\n\nMetrics for ${MODEL} modification ${mod_type}\n\n"
        uv run python src/calculate_metric.py \
            --results-dir ${RESULTS_DIR}/${MODEL}_${mod_type} \
            --exclude-confusion-matrix \
            --exclude-correlation
    done
done