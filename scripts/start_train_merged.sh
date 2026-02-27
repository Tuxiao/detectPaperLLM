export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export NUM_PERTURB_SAMPLES=8
export MAX_LENGTH=128
export TRAIN_FILE="../data/train_pairs_merged.jsonl"
export OUTPUT_DIR="../outputs/qwen3-0.6b-lora"
export NUM_TRAIN_EPOCHS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8
./run_qwen3_lora_example.sh
