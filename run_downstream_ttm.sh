#!/usr/bin/env bash
conda activate clamp3

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# 1. 准备文本文件
echo "[1/4] Preparing per-prompt text files..."
INPUT_PROMPT="InputData/ttm_regenerate/prompt_info.txt"
PROMPT_OUTDIR="InputData/ttm_regenerate/prompt_info"
python3 InputData/ttm/prepare_text_input.py --input "$INPUT_PROMPT" --output "$PROMPT_OUTDIR"

# 2. 提取文本 embedding（global）
echo "[2/4] Extracting text embeddings (global)..."
TEXT_EMBED_OUT="InputData/ttm_regenerate/text_embed"
python3 thirdparty/clamp3/clamp3_embd.py "$PROMPT_OUTDIR" "$TEXT_EMBED_OUT" --get_global

# 3. 提取音频 embedding（global）
echo "[3/4] Extracting audio embeddings (global)..."
WAV_DIR="InputData/ttm_regenerate/wavs"
AUDIO_EMBED_OUT="InputData/ttm_regenerate/audio_embed"
python3 thirdparty/clamp3/clamp3_embd.py "$WAV_DIR" "$AUDIO_EMBED_OUT" --get_global

# 4. 运行推理
echo "[4/4] Running downstream inference with MusicEval MosPredictor..."
OUTDIR="OutputData/ttm_eval"
CKPT_PATH="model_ckpt/musiceval_ckpt/best_ckpt_88"
mkdir -p "$OUTDIR"
python3 run_musiceval.py --outdir "$OUTDIR" --ckpt "$CKPT_PATH" --get_global \
	--audio_embed_dir "$AUDIO_EMBED_OUT" --text_embed_dir "$TEXT_EMBED_OUT"

echo "Done. Results are in $OUTDIR"
