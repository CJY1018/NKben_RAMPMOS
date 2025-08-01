#!/bin/bash

# 检查参数数量
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
export HF_ENDPOINT=https://hf-mirror.com # 国内镜像

# 创建临时的软链接到模型的配置和检查点目录，这样可以避免修改seed-vc的源代码
if [ "$MODEL_NAME" == "seed-vc" ]; then
  ln -s Upstream_VC/seed-vc/configs configs
  mkdir -p Upstream_VC/seed-vc/checkpoints
  ln -s Upstream_VC/seed-vc/checkpoints checkpoints
fi

for LANG in "${LANGUAGES[@]}"; do
  echo "正在调用 $MODEL_NAME 模型（$LANG）..."
  python Upstream_VC/run_$MODEL_NAME.py --lang "$LANG"
done

# 删除临时软链接
rm -f configs checkpoints

# conda activate seed-vc
# bash run_upstream_vc.sh seed-vc
