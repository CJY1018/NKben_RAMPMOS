"""
批量重采样音频文件为 16kHz。
支持常见格式 (.wav, .mp3, .flac)，并自动跳过已是16kHz的文件。
"""

import os
import argparse
import torchaudio

def resample_audio(in_path, out_path, target_sr=16000):
    """将单个音频文件重采样为 target_sr。"""
    try:
        waveform, sr = torchaudio.load(in_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torchaudio.save(out_path, waveform, target_sr)
        print(f"✅ {in_path} -> {out_path} ({target_sr} Hz)")
    except Exception as e:
        print(f"❌ Failed to process {in_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default = "InputData/ttm/wavs")
    parser.add_argument("--output_dir", type=str, default = "InputData/ttm/wavs")
    parser.add_argument("--sr", type=int, default=16000, help="目标采样率")
    args = parser.parse_args()

    # 遍历目录
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith((".wav", ".mp3", ".flac")):
                in_path = os.path.join(root, f)
                rel_path = os.path.relpath(in_path, args.input_dir)
                out_path = os.path.join(args.output_dir, rel_path)
                resample_audio(in_path, out_path, args.sr)

if __name__ == "__main__":
    main()
