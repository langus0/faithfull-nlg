uv run python src/eval_mod.py \
	--model eval_nemo \
	--template src/templates/zero_shot/qags.jinja \
	--data data/meta_eval/qags.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_nemo_severity-1_testt \
	--eval-mod severity \
	--mod-force -1 \
	--limit 3

