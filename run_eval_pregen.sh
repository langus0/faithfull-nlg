uv run python src/eval_mod.py \
	--model eval_nemo \
	--template src/templates/zero_shot/qags.jinja \
	--pregen data/results/pregen_results/qags-textsev/pregen_eval_nemo_textsev-1.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_nemo_int_and_textsev-1 \
	--eval-mod severity
