uv run python src/eval_zero_shot.py \
	--model eval_nemo \
	--template src/templates/zero_shot/qags.jinja \
	--data data/meta_eval/qags.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_nemo_severity-2_full_repeated \
	--eval-mod severity
