# MODEL=eval_nemo
MODEL=eval_gemma
# MODEL=eval_qwen
# MODEL=eval_mistral

# DATASET=qags
DATASET=hanna
# DATASET=summeval

# ASPECT=factual_consistency
ASPECT=coherence
# ASPECT=relevance
# ASPECT=complexity

PERTURBATION_MODEL=eval_gemma

TEMPLATE_PATH=src/templates/zero_shot/${DATASET}.jinja
ASPECT_PATH=src/configs/eval_aspects/${DATASET}-${ASPECT}.json

PREGEN_DIR=results/pregen_results/perturbed/by_${PERTURBATION_MODEL}
RESULTS_DIR=results/eval_mod_results/${DATASET}/perturbed_by_${ASPECT}/using_${PERTURBATION_MODEL}

  
# int severity (using a previously pregenerated evaluation)
uv run python src/eval_mod.py \
    --model ${MODEL} \
    --template ${TEMPLATE_PATH} \
    --aspect-config  ${ASPECT_PATH}\
    --data ${PREGEN_DIR}/${DATASET}-${ASPECT}.json \
    --output-dir ${RESULTS_DIR}/eval_${MODEL} \
    --eval-mod none
