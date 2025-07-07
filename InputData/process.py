import os
import shutil
import csv
from pathlib import Path

def process_audio_files():
    """
    读取zh和en目录下的meta.csv文件，提取音频路径并复制到新文件夹
    """
    # 定义基础路径
    base_path = Path(__file__).parent
    zh_path = base_path / "zh"
    en_path = base_path / "en"
    
    # 创建新的音频文件夹
    zh_audio_dir = zh_path / "prompt-wavs-selected"
    en_audio_dir = en_path / "prompt-wavs-selected"
    
    # 确保目标文件夹存在
    zh_audio_dir.mkdir(exist_ok=True)
    en_audio_dir.mkdir(exist_ok=True)
    
    # 处理中文音频文件
    print("正在处理中文音频文件...")
    process_language_files(zh_path, zh_audio_dir, "zh")
    
    # 处理英文音频文件
    print("正在处理英文音频文件...")
    process_language_files(en_path, en_audio_dir, "en")
    
    print("音频文件处理完成！")

def process_language_files(lang_path, audio_dir, lang_name):
    """
    处理指定语言的音频文件
    
    Args:
        lang_path: 语言目录路径
        audio_dir: 目标音频目录路径
        lang_name: 语言名称（用于日志输出）
    """
    meta_file = lang_path / "meta.csv"
    
    if not meta_file.exists():
        print(f"警告: {lang_name} 目录下未找到 meta.csv 文件")
        return
    
    copied_count = 0
    missing_count = 0
    
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 使用 | 分隔符分割行
                fields = line.split('|')
                if len(fields) < 3:
                    print(f"警告: 第 {line_num} 行字段数量不足: {line}")
                    continue
                
                # 获取第三个字段（音频路径）
                audio_path_str = fields[2]
                
                # 构建完整的源文件路径
                source_audio_path = lang_path / audio_path_str
                
                # 获取文件名
                audio_filename = Path(audio_path_str).name
                
                # 构建目标文件路径
                target_audio_path = audio_dir / audio_filename
                
                # 检查源文件是否存在
                if source_audio_path.exists():
                    try:
                        # 复制文件
                        shutil.copy2(source_audio_path, target_audio_path)
                        copied_count += 1
                        print(f"已复制: {audio_filename}")
                    except Exception as e:
                        print(f"复制文件失败 {audio_filename}: {e}")
                else:
                    missing_count += 1
                    print(f"文件不存在: {source_audio_path}")
        
        print(f"{lang_name} 处理完成:")
        print(f"  成功复制: {copied_count} 个文件")
        print(f"  缺失文件: {missing_count} 个")
        print(f"  目标目录: {audio_dir}")
        
    except Exception as e:
        print(f"处理 {lang_name} 文件时出错: {e}")

if __name__ == "__main__":
    process_audio_files()
