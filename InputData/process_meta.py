#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理meta.csv文件，删除每行最后一个|符号及其后的内容
"""

import os
import shutil

def process_meta_file(file_path):
    """
    处理单个meta.csv文件，删除每行最后一个|符号及其后的内容
    
    Args:
        file_path (str): meta.csv文件的路径
    """
    print(f"正在处理文件: {file_path}")
    
    # 读取原文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行
    processed_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # 找到最后一个|的位置
            last_pipe_index = line.rfind('|')
            if last_pipe_index != -1:
                # 删除最后一个|及其后的内容
                processed_line = line[:last_pipe_index]
                processed_lines.append(processed_line)
            else:
                # 如果没有找到|，保持原行不变
                processed_lines.append(line)
    
    # 创建备份文件
    backup_path = file_path + '.backup'
    shutil.copy2(file_path, backup_path)
    print(f"已创建备份文件: {backup_path}")
    
    # 写入处理后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')
    
    print(f"已处理完成: {file_path}")
    print(f"处理前行数: {len(lines)}, 处理后行数: {len(processed_lines)}")

def main():
    """主函数"""
    # 定义要处理的目录
    directories = ['InputData/zh', 'InputData/en']
    
    for directory in directories:
        meta_file = os.path.join(directory, 'meta.csv')
        
        if os.path.exists(meta_file):
            print(f"\n{'='*50}")
            print(f"处理目录: {directory}")
            print(f"{'='*50}")
            process_meta_file(meta_file)
        else:
            print(f"文件不存在: {meta_file}")
    
    print(f"\n{'='*50}")
    print("所有文件处理完成！")
    print("注意：原文件已备份为 .backup 文件")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
