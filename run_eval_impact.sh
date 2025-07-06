uv run python src/eval_mod_impact.py \
	--model eval_gemma \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_eval_gemma.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_gemma_impact-1 \
	--mod-direction 1 \

uv run python src/eval_mod_impact.py \
	--model eval_gemma \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_eval_gemma.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_gemma_impact-1 \
	--mod-direction -1 \