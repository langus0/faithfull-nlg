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

severity_modification_directions=(1 -1)

TEMPLATE_PATH=src/templates/zero_shot/${DATASET}.jinja
ASPECT_PATH=src/configs/eval_aspects/${DATASET}-${ASPECT}.json
PREGEN_DIR=results/pregen_results/${DATASET}/${ASPECT}
RESULTS_DIR=results/eval_mod_results/${DATASET}/${ASPECT}

for sev_dir in "${severity_modification_directions[@]}"
do
	echo "Running modifications impacts with severity direction: $sev_dir"

	uv run python src/eval_mod_per_error.py \
		--model ${MODEL} \
		--template ${TEMPLATE_PATH} \
		--aspect-config ${ASPECT_PATH} \
		--data ${PREGEN_DIR}/pregen_${MODEL}.json \
		--output-dir ${RESULTS_DIR}/${MODEL}_impact${sev_dir} \
		--per_error_mod impact
		--mod-direction ${sev_dir}

done
