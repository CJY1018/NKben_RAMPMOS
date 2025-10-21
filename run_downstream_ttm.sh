#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# 1. 准备文本文件
echo "[1/4] Preparing per-prompt text files..."
INPUT_PROMPT="InputData/ttm/prompt_info.txt"
PROMPT_OUTDIR="InputData/ttm/prompt_info"
python3 InputData/ttm/prepare_text_input.py --input "$INPUT_PROMPT" --output "$PROMPT_OUTDIR"

# 2. 提取文本 embedding（global）
echo "[2/4] Extracting text embeddings (global)..."
TEXT_EMBED_OUT="InputData/ttm/text_embed"
mkdir -p "$TEXT_EMBED_OUT"
python3 thirdparty/clamp3/clamp3_embd.py "$PROMPT_OUTDIR" "$TEXT_EMBED_OUT" --get_global

# 3. 提取音频 embedding（global）
echo "[3/4] Extracting audio embeddings (global)..."
WAV_DIR="InputData/ttm/wavs"
AUDIO_EMBED_OUT="InputData/ttm/audio_embed"
mkdir -p "$AUDIO_EMBED_OUT"
python3 thirdparty/clamp3/clamp3_embd.py "$WAV_DIR" "$AUDIO_EMBED_OUT" --get_global

# 4. 运行推理
echo "[4/4] Running downstream inference with MusiEval predictor..."
OUTDIR="OutputData/ttm_eval"
CKPT_PATH="model_ckpt/musiceval_ckpt/best_ckpt_28"
mkdir -p "$OUTDIR"
python3 run_musiceval.py --wavdir "$WAV_DIR" --outdir "$OUTDIR" --ckpt "$CKPT_PATH" --get_global \
	--audio_embed_dir "$AUDIO_EMBED_OUT" --text_embed_dir "$TEXT_EMBED_OUT"

echo "Done. Results are in $OUTDIR"
