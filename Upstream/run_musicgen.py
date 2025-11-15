"""
生成音乐文件的脚本，基于 Meta Audiocraft 的 MusicGen 模型。
"""
import os
import csv
import argparse
import torch
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str, default='InputData/ttm_regenerate/prompt_info.txt', help='prompt_info file path')
    parser.add_argument('--out_dir', type=str, required = True, help='output wav directory')
    parser.add_argument('--model', type=str, default='small', help='MusicGen model size: small | medium | large | melody')
    parser.add_argument('--duration', type=float, default=30, help='duration of generated music in seconds')
    parser.add_argument('--device', type=str, default=None, help='device to run model on (e.g. "cuda" or "cpu")')
    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f'prompt file not found: {args.prompt_file}')

    os.makedirs(args.out_dir, exist_ok=True)

    # 选择运行设备
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading MusicGen model: {args.model} on {device} ...')

    # 加载模型
    model = MusicGen.get_pretrained(args.model)

    # 改为确定性采样
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.set_generation_params(duration=args.duration)

    # 读取 prompt 文件
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)

    print(f'Found {len(rows)} prompts, generating audio into: {args.out_dir}')

    batch_size = 4
    i = 0
    total = len(rows)

    while i < total:
        batch = rows[i:i + batch_size]
        texts = []
        ids = []
        for row in batch:
            pid = row.get('id')
            txt = row.get('text')
            if not txt:
                print(f'Skipping {pid}: empty text')
                continue
            ids.append(pid)
            texts.append(txt)

        if not texts:
            i += batch_size
            continue

        out_names = [f'S010_{pid}.wav' for pid in ids]
        print(f'Generating batch {i // batch_size + 1}: {[str(p) for p in ids]} ...')

        try:
            with torch.no_grad():
                wavs = model.generate(texts)  # batch generation, len(wavs) == len(texts)

            # 保存每个样本
            for j, wav in enumerate(wavs):
                out_name = out_names[j]
                out_path = os.path.join(args.out_dir, out_name)
                audio_write(
                    stem_name=out_path.replace('.wav', ''),
                    wav=wav.cpu(),
                    sample_rate=model.sample_rate,
                )
                print(f'Saved: {out_path}')

        except Exception as e:
            print(f'Failed to generate batch starting at index {i}: {e}')

        i += batch_size
    
    print('All done.')


if __name__ == '__main__':
    main()
