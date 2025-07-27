### STEP 1: Choose model

# MODEL=eval_nemo
MODEL=eval_gemma
# MODEL=eval_qwen
# MODEL=eval_mistral


### STEP 2: Choose dataset & set of aspects

# DATASET=qags
# aspects=(factual_consistency)

DATASET=hanna
aspects=(coherence complexity relevance)

# DATASET=summeval
# aspects=(coherence factual_consistency relevance)


### STEP 3: Modification-specific parameters

#  0 = no deletion cascade and only deletion of current error
# -1 = deletion of current error and all previous ones
#  1 = deletion of current error and all further ones
# DELETION_CASCADE=-1
deletion_cascades=(0 1 -1)

for ASPECT in "${aspects[@]}"
do
	for DELETION_CASCADE in "${deletion_cascades[@]}"
	do
		TEMPLATE_PATH=src/templates/zero_shot/${DATASET}.jinja
		ASPECT_PATH=src/configs/eval_aspects/${DATASET}-${ASPECT}.json
		PREGEN_DIR=results2/pregen_results/${DATASET}/${ASPECT}
		RESULTS_DIR=results2/eval_mod_results/${DATASET}/${ASPECT}

		echo "Running error deletion ${DELETION_CASCADE} for $DATASET-$ASPECT using model $MODEL"

		uv run python src/eval_mod_per_error_vllm.py \
			--model ${MODEL} \
			--template ${TEMPLATE_PATH} \
			--aspect-config ${ASPECT_PATH} \
			--data ${PREGEN_DIR}/pregen_${MODEL}.json \
			--output-dir ${RESULTS_DIR}/${MODEL}_delete${DELETION_CASCADE} \
			--per_error_mod delete \
			--mod-direction ${DELETION_CASCADE}

	done
done
