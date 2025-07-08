MODEL=eval_gemma
ASPECT=src/configs/eval_aspects/qags-factual_consistency.json
TEMPLATE=src/templates/zero_shot/qags.jinja
severity_modification_forces=(1 2 -1 -2)

# if it is the first run for this set of OpeNLG parameters (model, aspect, template)
# then run the first evaluation to generate the initial pregen (for example with severity modification 1)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template ${TEMPLATE} \
	--aspect-config  ${ASPECT}\
	--data data/meta_eval/qags.json \
	--output-dir data/results/${MODEL}_severity1 \
	--eval-mod severity \
	--mod-force 1

# save the OpeNLG evaluation in a pregen file (without the severity modification)
# it will be then used by following scripts to avoid generating OpeNLG evaluation again
uv run python src/copy_results_to_pregen.py \
	--results-dir data/results/${MODEL}_severity1 \
	--pregen-dest-dir data/results/pregen_results/qags \
	--pregen-tag ${MODEL} \
	--exclude-premodified-result

for sev_force in "${severity_modification_forces[@]}"
do
	echo "Running modifications with severity: $sev_force"
  
	# int severity (using a previously pregenerated evaluation)
	uv run python src/eval_mod.py \
		--model ${MODEL} \
		--template ${TEMPLATE} \
		--aspect-config  ${ASPECT}\
		--data data/results/pregen_results/qags/pregen_${MODEL}.json \
		--output-dir data/results/${MODEL}_severity${sev_force} \
		--eval-mod severity \
		--mod-force ${sev_force}

	# text severity (using a previously pregenerated evaluation)
	uv run python src/eval_mod.py \
		--model ${MODEL} \
		--template ${TEMPLATE} \
		--aspect-config  ${ASPECT}\
		--data data/results/pregen_results/qags/pregen_${MODEL}.json \
		--output-dir data/results/${MODEL}_textsev${sev_force} \
		--eval-mod text_severity \
		--mod-force ${sev_force}

	# now create a pregen file based on the text modification, which will be used in the next joined modification
	uv run python src/copy_results_to_pregen.py \
		--results-dir data/results/${MODEL}_textsev${sev_force} \
		--pregen-dest-dir data/results/pregen_results/qags-textsev \
		--pregen-tag ${MODEL}_textsev${sev_force}

	# int and text severity -1
	# (using the newly pregenerated modification and its modified text of the results)
	uv run python src/eval_mod.py \
		--model ${MODEL} \
		--template ${TEMPLATE} \
		--aspect-config  ${ASPECT}\
		--data data/results/pregen_results/qags-textsev/pregen_${MODEL}_textsev${sev_force}.json \
		--output-dir data/results/${MODEL}_int_and_textsev${sev_force} \
		--eval-mod severity \
		--mod-force ${sev_force} \
		--use-premodified-result


done
