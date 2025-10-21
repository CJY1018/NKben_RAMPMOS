"""
生成音乐文件的脚本，基于Hugging Face的MusicGen模型。
"""
import os
import csv
import re
import argparse
import numpy as np
import scipy.io.wavfile
from transformers import pipeline

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--prompt_file', type=str, default = 'InputData/ttm/prompt_info.txt', help='prompt_info file path')
	parser.add_argument('--out_dir', type=str, default='InputData/ttm/wavs', help='output wav directory')
	parser.add_argument('--model', type=str, default='facebook/musicgen-small', help='model for text-to-music')
	parser.add_argument('--device', type=str, default=None, help='device for transformers pipeline (e.g. "cuda" or "cpu")')
	args = parser.parse_args()

	script_dir = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.dirname(script_dir)

	prompt_file = args.prompt_file or os.path.join(repo_root, 'InputData', 'ttm', 'prompt_info.txt')
	out_dir = args.out_dir or os.path.join(repo_root, 'InputData', 'ttm', 'wavs')
	
	if not os.path.exists(prompt_file):
		raise FileNotFoundError(f'prompt file not found: {prompt_file}')

	print(f'Loading text-to-music model: {args.model} ...')
	kwargs = {}
	if args.device:
		kwargs['device'] = args.device
	synthesiser = pipeline('text-to-audio', args.model, **kwargs)

	# read prompts (tab-separated file with header id \t text)
	with open(prompt_file, 'r', encoding='utf-8') as f:
		reader = csv.DictReader(f, delimiter='\t')
		rows = list(reader)

	print(f'Found {len(rows)} prompts, generating audio into: {out_dir}')

	for row in rows:
		prompt_id = row.get('id')
		text = row.get('text')
		out_name = f'S010_{prompt_id}.wav'
		out_path = os.path.join(out_dir, out_name)
		if not text:
			print(f'Skipping {prompt_id}: empty text')
			continue
		try:
			print(f'Generating {out_name} ...')
			music = synthesiser(text, forward_params={"do_sample": True})
			audio = music.get('audio') if isinstance(music, dict) else music
			sr = music.get('sampling_rate', 16000) if isinstance(music, dict) else 16000

			# ensure numpy array
			audio_np = np.asarray(audio)
			# convert float audio in [-1,1] to int16 if needed
			if np.issubdtype(audio_np.dtype, np.floating):
				audio_int16 = (audio_np * 32767).astype(np.int16)
			else:
				audio_int16 = audio_np

			scipy.io.wavfile.write(out_path, rate=sr, data=audio_int16)
			print(f'Saved: {out_path}')
		except Exception as e:
			print(f'Failed to generate for {prompt_id}: {e}')

	print('All done.')


if __name__ == '__main__':
	main()