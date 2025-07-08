#!/bin/bash

# 检查参数数量
if [ "$#" -lt 1 ]; then
  echo "用法: $0 <模型名称>"
  exit 1
fi

MODEL_NAME="$1"

# 定义语言列表
LANGUAGES=("en" "zh")

if [ "$MODEL_NAME" != "cosyvoice2" ] && [ "$MODEL_NAME" != "xtts" ]; then
  echo "错误：未知的模型名称 '$MODEL_NAME'"
  echo "请使用 'cosyvoice2' 或 'xtts' 或自定义 'example'"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:Upstream/CosyVoice/third_party/Matcha-TTS

for LANG in "${LANGUAGES[@]}"; do
  echo "正在调用 $MODEL_NAME 模型（$LANG）..."
  python Upstream/run_$MODEL_NAME.py --lang "$LANG"
done

# conda activate cosyvoice
# bash run_upstream.sh cosyvoice2

# conda activate xtts
# bash run_upstream.sh xtts