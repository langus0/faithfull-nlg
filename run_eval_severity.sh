uv run python src/eval_mod.py \
	--model eval_nemo \
	--template src/templates/zero_shot/qags.jinja \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--data data/meta_eval/qags.json \
	--output-dir results/eval_mod_results/qags/factual_consistency/eval_nemo_severity1_test \
	--eval-mod severity \
	--mod-force 1 \
	--limit 3

