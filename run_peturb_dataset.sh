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

ASPECT_PATH=src/configs/eval_aspects/${DATASET}-${ASPECT}.json

uv run python  -m src.perturb_dataset \
    --dataset-name ${DATASET} \
    --model ${MODEL} \
    --aspect ${ASPECT}
