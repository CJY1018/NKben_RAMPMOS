#!/bin/bash

# 检查是否传入参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <位置>"
    exit 1
fi

# 接收输入的位置
POSITION=$1

# 构造 meta.csv 的路径
META_CSV="$POSITION/meta.csv"

# 执行 Python 脚本
python run_all.py --meta_csv "$META_CSV"
