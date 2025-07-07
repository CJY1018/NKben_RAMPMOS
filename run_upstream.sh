#!/bin/bash

# 检查参数数量
if [ "$#" -lt 2 ]; then
  echo "用法: $0 <模型名称> <输出路径> [输入路径]"
  exit 1
fi

MODEL_NAME="$1"
OUTPUT_PATH="$2"
INPUT_PATH="${3:-InputData}"  # 默认输入路径为 InputData

echo "模型名称: $MODEL_NAME"
echo "输出路径: $OUTPUT_PATH"
echo "输入路径: $INPUT_PATH"

# 定义语言列表
LANGUAGES=("en" "zh")

if [ "$MODEL_NAME" == "cosyvoice" ]; then
  for LANG in "${LANGUAGES[@]}"; do
    echo "正在调用 CosyVoice 模型（$LANG）..."
    python run_cosyvoice.py --input "$INPUT_PATH/$LANG" --output "$OUTPUT_PATH/$LANG"
  done

elif [ "$MODEL_NAME" == "example" ]; then
  for LANG in "${LANGUAGES[@]}"; do
    echo "正在调用 example 模型（$LANG）..."
    python Upstream/run_example.py --input "$INPUT_PATH/$LANG" --output "$OUTPUT_PATH/$LANG"
  done

else
  echo "错误：未知的模型名称 '$MODEL_NAME'"
  echo "请使用 'cosyvoice' 或 'example'"
  exit 1
fi
