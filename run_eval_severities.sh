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
DATASET_PATH=data/meta_eval/${DATASET}.json
RESULTS_DIR=results/eval_mod_results/${DATASET}/${ASPECT}
PREGEN_DIR=results/pregen_results/${DATASET}/${ASPECT}

# if it is the first run for this set of OpeNLG parameters (model, aspect, template)
# then run the first evaluation to generate the initial pregen (for example with severity modification 1)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template ${TEMPLATE_PATH} \
	--aspect-config  ${ASPECT_PATH}\
	--data ${DATASET_PATH} \
	--output-dir ${RESULTS_DIR}/${MODEL} \
	--eval-mod none

# save the OpeNLG evaluation in a pregen file (without the severity modification)
# it will be then used by following scripts to avoid generating OpeNLG evaluation again
uv run python src/copy_results_to_pregen.py \
	--results-dir ${RESULTS_DIR}/${MODEL} \
	--pregen-dest-dir ${PREGEN_DIR} \
	--pregen-tag ${MODEL} \
	--exclude-premodified-result

# models=(eval_nemo eval_gemma)
aspects=(coherence complexity relevance)

severity_modification_forces=(1 2 -1 -2)

for ASPECT in "${aspects[@]}"
do
	for sev_force in "${severity_modification_forces[@]}"
	do
		echo "Running modifications with severity: $sev_force"
	
		# int severity (using a previously pregenerated evaluation)
		uv run python src/eval_mod.py \
			--model ${MODEL} \
			--template ${TEMPLATE_PATH} \
			--aspect-config  ${ASPECT_PATH}\
			--data ${PREGEN_DIR}/pregen_${MODEL}.json \
			--output-dir ${RESULTS_DIR}/${MODEL}_severity${sev_force} \
			--eval-mod severity \
			--mod-force ${sev_force}


		# text severity (using a previously pregenerated evaluation)
		uv run python src/eval_mod.py \
			--model ${MODEL} \
			--template ${TEMPLATE_PATH} \
			--aspect-config  ${ASPECT_PATH}\
			--data ${PREGEN_DIR}/pregen_${MODEL}.json \
			--output-dir ${RESULTS_DIR}/${MODEL}_textsev${sev_force} \
			--eval-mod text_severity \
			--mod-force ${sev_force}


		# now create a pregen file based on the text modification, which will be used in the next joined modification
		uv run python src/copy_results_to_pregen.py \
			--results-dir ${RESULTS_DIR}/${MODEL}_textsev${sev_force} \
			--pregen-dest-dir ${PREGEN_DIR} \
			--pregen-tag ${MODEL}_textsev${sev_force}

		# int and text severity -1
		# (using the newly pregenerated modification and its modified text of the results)
		uv run python src/eval_mod.py \
			--model ${MODEL} \
			--template ${TEMPLATE_PATH} \
			--aspect-config  ${ASPECT_PATH}\
			--data ${PREGEN_DIR}/pregen_${MODEL}_textsev${sev_force}.json \
			--output-dir ${RESULTS_DIR}/${MODEL}_int_and_textsev${sev_force} \
			--eval-mod severity \
			--mod-force ${sev_force} \
			--use-premodified-result

	done
done
