# #### text severity +1
# uv run python src/eval_mod.py \
# 	--model eval_gemma \
# 	--template src/templates/zero_shot/qags.jinja \
# 	--data data/meta_eval/qags.json \
# 	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
# 	--output-dir data/results/eval_gemma_textsev1 \
# 	--eval-mod text_severity \
# 	--mod-force 1 \

# uv run python src/copy_results_to_pregen.py \
# 	--results-path data/results/eval_gemma_textsev1 \
# 	--pregen-dest-path data/results/pregen_results/qags-textsev \
# 	--model eval_gemma_textsev1

# #### +1

# # int severity +1 (using pregenerated evaluation but not the modified text of the results)
# uv run python src/eval_mod.py \
# 	--model eval_gemma \
# 	--template src/templates/zero_shot/qags.jinja \
# 	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev1.json \
# 	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
# 	--output-dir data/results/eval_gemma_severity1 \
# 	--eval-mod severity \
# 	--mod-force 1 \

# # int and text severity +1 (using the modified text of the results)
# uv run python src/eval_mod.py \
# 	--model eval_gemma \
# 	--template src/templates/zero_shot/qags.jinja \
# 	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev1.json \
# 	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
# 	--output-dir data/results/eval_gemma_int_and_textsev1 \
# 	--eval-mod severity \
# 	--mod-force 1 \
# 	--use-premodified-result \

#### +2

# # int severity +2
# # (using a previously pregenerated evaluation but not its modified text of the results)
# uv run python src/eval_mod.py \
# 	--model eval_gemma \
# 	--template src/templates/zero_shot/qags.jinja \
# 	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev1.json \
# 	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
# 	--output-dir data/results/eval_gemma_severity2 \
# 	--eval-mod severity \
# 	--mod-force 2 \

# create pregenerated modification with text severity +2
# (using a previously pregenerated evaluation but not its modified text of the results)
uv run python src/eval_mod.py \
	--model eval_gemma \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev1.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_gemma_textsev2 \
	--eval-mod text_severity \
	--mod-force 2 \

uv run python src/copy_results_to_pregen.py \
	--results-path data/results/eval_gemma_textsev2 \
	--pregen-dest-path data/results/pregen_results/qags-textsev \
	--model eval_gemma_textsev2

# int and text severity +2 (using the modified text of the results)
uv run python src/eval_mod.py \
	--model eval_gemma \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev2.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_gemma_int_and_textsev2 \
	--eval-mod severity \
	--mod-force 2 \
	--use-premodified-result \

#### -1

# # int severity -1
# # (using a previously pregenerated evaluation but not its modified text of the results)
# uv run python src/eval_mod.py \
# 	--model eval_gemma \
# 	--template src/templates/zero_shot/qags.jinja \
# 	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev1.json \
# 	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
# 	--output-dir data/results/eval_gemma_severity-1 \
# 	--eval-mod severity \
# 	--mod-force -1 \

# create pregenerated modified text with text severity -1
# (using a previously pregenerated evaluation but not its modified text of the results)
uv run python src/eval_mod.py \
	--model eval_gemma \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev1.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_gemma_textsev-1 \
	--eval-mod text_severity \
	--mod-force -1 \

uv run python src/copy_results_to_pregen.py \
	--results-path data/results/eval_gemma_textsev-1 \
	--pregen-dest-path data/results/pregen_results/qags-textsev \
	--model eval_gemma_textsev-1

# int and text severity -1 (using the modified text of the results)
uv run python src/eval_mod.py \
	--model eval_gemma \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev-1.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_gemma_int_and_textsev-1 \
	--eval-mod severity \
	--mod-force -1 \
	--use-premodified-result \

#### -2

# # int severity -2
# # (using a previously pregenerated evaluation but not its modified text of the results)
# uv run python src/eval_mod.py \
# 	--model eval_gemma \
# 	--template src/templates/zero_shot/qags.jinja \
# 	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev1.json \
# 	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
# 	--output-dir data/results/eval_gemma_severity-2 \
# 	--eval-mod severity \
# 	--mod-force -2 \

# create pregenerated modified text with text severity -2 
# (using a previously pregenerated evaluation but not its modified text of the results)
uv run python src/eval_mod.py \
	--model eval_gemma \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev1.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_gemma_textsev-2 \
	--eval-mod text_severity \
	--mod-force -2 \

uv run python src/copy_results_to_pregen.py \
	--results-path data/results/eval_gemma_textsev-2 \
	--pregen-dest-path data/results/pregen_results/qags-textsev \
	--model eval_gemma_textsev-2

# int and text severity -2 (using the modified text of the results)
uv run python src/eval_mod.py \
	--model eval_gemma \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_eval_gemma_textsev-2.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/eval_gemma_int_and_textsev-2 \
	--eval-mod severity \
	--mod-force -2 \
	--use-premodified-result \
