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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str, default='InputData/ttm/prompt_info.txt', help='prompt_info file path')
    parser.add_argument('--out_dir', type=str, default='InputData/ttm/wavs', help='output wav directory')
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
    model.set_generation_params(duration=args.duration)
    model = model.to(device)

    # 读取 prompt 文件
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)

    print(f'Found {len(rows)} prompts, generating audio into: {args.out_dir}')

    for row in rows:
        prompt_id = row.get('id')
        text = row.get('text')

        if not text:
            print(f'Skipping {prompt_id}: empty text')
            continue

        out_name = f'S010_{prompt_id}.wav'
        out_path = os.path.join(args.out_dir, out_name)

        try:
            print(f'Generating {out_name} from text: "{text}" ...')

            with torch.no_grad():
                wav = model.generate([text])  # batch size = 1

            # 取出第一个样本并保存
            audio_write(
                out_path.replace('.wav', ''),  # audio_write会自动加上.wav
                wav[0].cpu(),
                model.sample_rate,
                strategy="loudness"
            )

            print(f'Saved: {out_path}')

        except Exception as e:
            print(f'Failed to generate for {prompt_id}: {e}')

    print('All done.')


if __name__ == '__main__':
    main()
