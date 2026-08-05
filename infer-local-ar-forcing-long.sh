# mkdir -p checkpoints/liveedit
# huggingface-cli download cp-cp/LiveEdit ar-forcing_002000.pt \
#   --local-dir checkpoints/liveedit

CKPT_PATH="checkpoints/liveedit/ar-forcing_002000.pt"

# 12-frame logical window = 3 persistent sink frames + 9 most recent frames.
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-12}"
SINK_SIZE="${SINK_SIZE:-3}"

CUDA_VISIBLE_DEVICES=0 python inference-mm.py \
    --config_path configs/wan_mm-ar-forcing-local.yaml \
    --output_folder "videos/long_test-120-wo-wrap" \
    --checkpoint_path "${CKPT_PATH}" \
    --data_path "./test_cases/long_test.json" \
    --num_output_frames 120 \
    --local_attn_size "${LOCAL_ATTN_SIZE}" \
    --sink_size "${SINK_SIZE}" \
    --window_rope \
    --task v2v