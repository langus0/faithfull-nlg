### STEP 1: Choose model

# MODEL=eval_nemo
MODEL=eval_gemma
# MODEL=eval_qwen


### STEP 2: Choose dataset & set of aspects

# DATASET=qags
# aspects=(factual_consistency)

DATASET=hanna
aspects=(coherence complexity relevance)

# DATASET=summeval
# aspects=(coherence factual_consistency relevance)


### STEP 3: Modification-specific parameters

numbers_of_errors=(1 2)


for ASPECT in "${aspects[@]}"
do
	TEMPLATE_PATH=src/templates/zero_shot/${DATASET}.jinja
	DATASET_PATH=data/meta_eval/${DATASET}.json
	RESULTS_DIR=results/eval_mod_results/${DATASET}/${ASPECT}
	PREGEN_DIR=results/pregen_results/${DATASET}/${ASPECT}
	ASPECT_PATH=src/configs/eval_aspects/${DATASET}-${ASPECT}.json

	for NUMBER_OF_ERRORS in "${numbers_of_errors[@]}"
	do
		echo "Adding $NUMBER_OF_ERRORS errors on $DATASET-$ASPECT for model: $MODEL"

		uv run python src/eval_mod.py \
			--model ${MODEL} \
			--template ${TEMPLATE_PATH} \
			--aspect-config  ${ASPECT_PATH}\
			--data ${PREGEN_DIR}/pregen_${MODEL}.json \
			--output-dir ${RESULTS_DIR}/${MODEL}_add_random_error${NUMBER_OF_ERRORS} \
			--eval-mod add_random_error \
			--mod-force ${NUMBER_OF_ERRORS}
	done
done
