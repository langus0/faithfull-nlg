#MODEL=eval_nemo
# MODEL=eval_gemma
# MODEL=eval_qwen
models=(eval_nemo eval_gemma eval_qwen)

DATASET=qags
# DATASET=hanna
# DATASET=summeval

aspects=(factual_consistencyy)
# aspects=(coherence complexity relevance)
# aspects=(coherence factual_consistency relevance)

numbers_of_errors=(1 2)


for MODEL in "${models[@]}"
do
	for ASPECT in "${aspects[@]}"
	do
		TEMPLATE_PATH=src/templates/zero_shot/${DATASET}.jinja
		DATASET_PATH=data/meta_eval/${DATASET}.json
		RESULTS_DIR=results2/eval_mod_results/${DATASET}/${ASPECT}
		PREGEN_DIR=results2/pregen_results/${DATASET}/${ASPECT}
		ASPECT_PATH=src/configs/eval_aspects/${DATASET}-${ASPECT}.json

		for NUMBER_OF_ERRORS in "${numbers_of_errors[@]}"
		do
			echo "Adding $NUMBER_OF_ERRORS errors on $DATASET-$ASPECT for model: $MODEL"

			uv run python src/eval_mod_vllm.py \
				--model ${MODEL} \
				--template ${TEMPLATE_PATH} \
				--aspect-config  ${ASPECT_PATH}\
				--data ${PREGEN_DIR}/pregen_${MODEL}.json \
				--output-dir ${RESULTS_DIR}/${MODEL}_add_random_error${NUMBER_OF_ERRORS} \
				--eval-mod add_random_error \
				--mod-force ${NUMBER_OF_ERRORS}
		done
	done
done