# MODEL=eval_nemo
MODEL=eval_gemma
# MODEL=eval_qwen
# MODEL=eval_mistral

# DATASET=qags
DATASET=hanna
# DATASET=summeval

# ASPECT=factual_consistency
# ASPECT=coherence
# ASPECT=relevance
# ASPECT=complexity
aspects=(coherence complexity relevance)

# set to -1 for deletion of current error and all previous ones
# set to 1 for deletion of current error and all further ones
# set to 0 for no deletion cascade and only deletion of current error
# DELETION_CASCADE=-1
deletion_cascades=(1 -1)

for DELETION_CASCADE in "${deletion_cascades[@]}"
do
	for ASPECT in "${aspects[@]}"
	do
		TEMPLATE_PATH=src/templates/zero_shot/${DATASET}.jinja
		ASPECT_PATH=src/configs/eval_aspects/${DATASET}-${ASPECT}.json
		PREGEN_DIR=results2/pregen_results/${DATASET}/${ASPECT}
		RESULTS_DIR=results2/eval_mod_results/${DATASET}/${ASPECT}

		echo "Running error deletion tests ${DELETION_CASCADE} for aspect: $ASPECT"
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