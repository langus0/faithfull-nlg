MODEL=eval_gemma

#### +1

# int severity +1
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/meta_eval/qags.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_severity1 \
	--eval-mod severity \
	--mod-force 1 \

# save the basic evaluation in a pregen file (without the modification), which will be then reused by all following scripts
uv run python src/copy_results_to_pregen.py \
	--results-path data/results/${MODEL}_severity1 \
	--pregen-dest-path data/results/pregen_results/qags \
	--model ${MODEL} \
	--exclude-premodified-result

# text severity +1 (using a previously pregenerated evaluation)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_${MODEL}.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_textsev1 \
	--eval-mod text_severity \
	--mod-force 1 \

# now create a pregen file based on previous modification for the next joined modification
uv run python src/copy_results_to_pregen.py \
	--results-path data/results/${MODEL}_textsev1 \
	--pregen-dest-path data/results/pregen_results/qags-textsev \
	--model ${MODEL}_textsev1

# int and text severity +1
# (using the newly pregenerated modification and its modified text of the results)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_${MODEL}_textsev1.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_int_and_textsev1 \
	--eval-mod severity \
	--mod-force 1 \
	--use-premodified-result \





#### +2

# int severity +2
# (using a previously pregenerated evaluation)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_${MODEL}.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_severity2 \
	--eval-mod severity \
	--mod-force 2 \

# text severity +2
# create a pregenerated modification (using a previously pregenerated evaluation)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_${MODEL}.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_textsev2 \
	--eval-mod text_severity \
	--mod-force 2 \

# now create a pregen file based on previous modification for the next joined modification
uv run python src/copy_results_to_pregen.py \
	--results-path data/results/${MODEL}_textsev2 \
	--pregen-dest-path data/results/pregen_results/qags-textsev \
	--model ${MODEL}_textsev2

# int and text severity +2
# (using the newly pregenerated modification and its modified text of the results)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_${MODEL}_textsev2.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_int_and_textsev2 \
	--eval-mod severity \
	--mod-force 2 \
	--use-premodified-result \





#### -1

# int severity -1 (using a previously pregenerated evaluation)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_${MODEL}.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_severity-1 \
	--eval-mod severity \
	--mod-force -1 \

# text severity -1 (using a previously pregenerated evaluation)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_${MODEL}.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_textsev-1 \
	--eval-mod text_severity \
	--mod-force -1 \

# now create a pregen file based on previous modification for the next joined modification
uv run python src/copy_results_to_pregen.py \
	--results-path data/results/${MODEL}_textsev-1 \
	--pregen-dest-path data/results/pregen_results/qags-textsev \
	--model ${MODEL}_textsev-1

# int and text severity -1
# (using the newly pregenerated modification and its modified text of the results)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_${MODEL}_textsev-1.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_int_and_textsev-1 \
	--eval-mod severity \
	--mod-force -1 \
	--use-premodified-result \






#### -2

# int severity -2 (using a previously pregenerated evaluation)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_${MODEL}.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_severity-2 \
	--eval-mod severity \
	--mod-force -2 \

# text severity -2 (using a previously pregenerated evaluation)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags/pregen_${MODEL}.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_textsev-2 \
	--eval-mod text_severity \
	--mod-force -2 \

# now create a pregen file based on previous modification for the next joined modification
uv run python src/copy_results_to_pregen.py \
	--results-path data/results/${MODEL}_textsev-2 \
	--pregen-dest-path data/results/pregen_results/qags-textsev \
	--model ${MODEL}_textsev-2

# int and text severity -2
# (using the newly pregenerated modification and its modified text of the results)
uv run python src/eval_mod.py \
	--model ${MODEL} \
	--template src/templates/zero_shot/qags.jinja \
	--data data/results/pregen_results/qags-textsev/pregen_${MODEL}_textsev-2.json \
	--aspect-config src/configs/eval_aspects/qags-factual_consistency.json \
	--output-dir data/results/${MODEL}_int_and_textsev-2 \
	--eval-mod severity \
	--mod-force -2 \
	--use-premodified-result \
