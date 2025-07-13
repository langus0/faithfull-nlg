# MODEL=eval_nemo
MODEL=eval_gemma
# MODEL=eval_qwen
# MODEL=eval_mistral

DATASET=qags
# DATASET=hanna
# DATASET=summeval

ASPECT=factual_consistency
# ASPECT=coherence
# ASPECT=relevance
# ASPECT=complexity

TEMPLATE_PATH=src/templates/zero_shot/${DATASET}.jinja
ASPECT_PATH=src/configs/eval_aspects/${DATASET}-${ASPECT}.json
PREGEN_DIR=results/pregen_results/${DATASET}/${ASPECT}
RESULTS_DIR=results/eval_mod_results/${DATASET}/${ASPECT}

echo "Running error deletion tests"

uv run python src/eval_mod_per_error.py \
	--model ${MODEL} \
	--template ${TEMPLATE_PATH} \
	--aspect-config ${ASPECT_PATH} \
	--data ${PREGEN_DIR}/pregen_${MODEL}.json \
	--output-dir ${RESULTS_DIR}/${MODEL}_delete \
	--per_error_mod delete
	--mod-direction ${sev_dir}

