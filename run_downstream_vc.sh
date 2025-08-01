#!/bin/bash

# 检查是否传入参数
if [ "$#" -lt 1 ]; then
  echo "用法: $0 <模型名称>"
  exit 1
fi

MODEL_NAME="$1"

# 定义语言列表
LANGUAGES=("en" "zh")

if [ "$MODEL_NAME" != "seed-vc" ]; then
  echo "错误：未知的模型名称 '$MODEL_NAME'"
  echo "请使用 'seed-vc' 或 自定义 'example'"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

for LANG in "${LANGUAGES[@]}"; do
  echo "正在评估 $MODEL_NAME 模型（$LANG）..."
  python run_all_vc.py --meta_csv "InputData_VC/${LANG}/meta_$MODEL_NAME.csv" --lang "$LANG"
done

# conda activate eval
# bash run_downstream_vc.sh seed-vc