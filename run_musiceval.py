#!/usr/bin/env python3

import os
import argparse
import json
from pathlib import Path
import subprocess
import torch
from tqdm import tqdm
from model.musiceval_baseline import MosPredictor, load_clamp3_from_filename

def load_checkpoint(checkpoint_path, device, ssl_out_dim=768):

    model = MosPredictor(ssl_out_dim).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)

    return model


def run_inference(model, device, out_file, audio_embed_dir, text_embed_dir):
    wav_files = sorted([p for p in Path(audio_embed_dir).rglob('*.npy')])
    predict = {}
    for wav_name in tqdm(wav_files, desc='Inferring'):
        # p:S001_P001，据此加载对应的 audio/text embed
        audio_embeds, text_embeds = load_clamp3_from_filename([str(wav_name)], audio_embed_dir, text_embed_dir, device=device)

        with torch.no_grad():
            output = model(audio_embeds,text_embeds)

        out_val = output.cpu().numpy().squeeze().tolist()  # [mos1,mos2]
        wav_id = wav_name.split('.')[0]
        predict[wav_id] = float(out_val)

    # write txt and json
    txt_lines = []
    for wav_id in sorted(predict.keys()):
        txt_lines.append(f"{wav_id},{predict[wav_id]}\n")
    with open(out_file + '.txt', 'w') as f:
        f.writelines(txt_lines)
    with open(out_file + '.json', 'w') as f:
        json.dump(predict, f, indent=4)

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=False, default = 'OutputData/ttm_eval', help='Output directory for embeddings and results')
    parser.add_argument('--ckpt', required=False, default = 'model_ckpt/musiceval_ckpt/best_ckpt_28', help='Path to finetuned MLP checkpoint')
    parser.add_argument('--get_global', action='store_true', help='Extract global embeddings (1,dim)')
    parser.add_argument('--ssl_dim', type=int, default=768, help='SSL output dimension (default 768 for clamp3)')
    parser.add_argument('--audio_embed_dir', type=str, default=None, help='clamp3 audio embed')
    parser.add_argument('--text_embed_dir', type=str, default=None, help='clamp3 text embed')
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 1. load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading checkpoint...')
    model = load_checkpoint(args.ckpt, device, ssl_out_dim=args.ssl_dim)

    # 2. run inference
    print('Running inference...')
    out_file = os.path.join(outdir, 'results')

    predictions = run_inference(model, device, out_file, args.audio_embed_dir, args.text_embed_dir)
    print('Saved results to', out_file + '.txt and .json')

if __name__ == '__main__':
    main()
