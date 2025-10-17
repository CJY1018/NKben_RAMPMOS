#!/usr/bin/env python3
"""
Run full music evaluation pipeline:
1. extract CLAMP3 embeddings for an input wav folder or file-list
2. load a finetuned MosPredictor checkpoint
3. run inference and save MOS predictions

Usage:
    python run_musiceval.py --wavdir /path/to/wavs --outdir /path/to/out --ckpt /path/to/best_ckpt
    optional: --get_global to extract (1,768) global embeddings

This script reuses helper functions from clamp3_embd.py and mos_clamp_extract.py in the repo.
"""
import os
import sys
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model.clamp3_embd import extract_audio_features
from model.mos_clamp_extract import MosPredictor, load_clamp3_from_filename


def extract_embeddings(wavdir, embed_outdir, get_global=False):
    """Extract embeddings for wavdir into embed_outdir using the existing extractor.
    If embed_outdir exists, it will be overwritten.
    Returns the path to embed_outdir.
    """
    os.makedirs(embed_outdir, exist_ok=True)
    # call extract_audio_features from clamp3_embd which mirrors clamp3_embd.py behavior
    extract_audio_features(wavdir, embed_outdir, get_global)
    return embed_outdir


def load_checkpoint(checkpoint_path, device, ssl_out_dim=768):
    """Load MosPredictor and checkpoint.
    Returns (model, device)
    """
    model = MosPredictor(ssl_out_dim).to(device)
    model.eval()

    # checkpoint may be a torch.save'd state_dict
    ckpt = torch.load(checkpoint_path, map_location=device)
    # the example evaluation script used ckpt as state_dict directly
    try:
        model.load_state_dict(ckpt)
    except Exception:
        # try to find 'state_dict' or 'model' key
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            raise
    return model


def run_inference(embeds_dir, model, device, out_file):
    """Run inference over embeddings saved in embeds_dir.
    Assumes embeddings are saved in a format load_clamp3_from_filename can load given a list of filenames.
    Writes predictions to out_file (txt + json).
    """
    # find .npy files in embeds_dir
    files = sorted([p for p in Path(embeds_dir).rglob('*.npy')])
    predict = {}
    # process one-by-one to save memory (the evaluation script used batch_size=1)
    for p in tqdm(files, desc='Inferring'):
        # p is path to e.g. wavname.npy or wavname_global.npy depending on extractor
        fname = p.name
        # load embeddings via load_clamp3_from_filename if available
        try:
            audio_embeds, text_embeds = load_clamp3_from_filename([str(p)])
        except Exception:
            # fallback: direct load
            arr = np.load(str(p))
            # if global, expect shape (1,dim) or (dim,)
            if arr.ndim == 2 and arr.shape[0] == 1:
                audio_embeds = torch.tensor(arr).float().to(device)
            elif arr.ndim == 1:
                audio_embeds = torch.tensor(arr[None, ...]).float().to(device)
            else:
                audio_embeds = torch.tensor(arr).float().to(device)
            text_embeds = None

        with torch.no_grad():
            if text_embeds is None:
                out = model(audio_embeds)
                # model may return tuple for multi-task; take first output
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
            else:
                out = model(audio_embeds, text_embeds)
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]

        out_val = out.cpu().numpy().squeeze().tolist()
        wav_id = fname.split('.')[0]
        predict[wav_id] = float(out_val)

    # write txt and json
    txt_lines = []
    for k in sorted(predict.keys()):
        txt_lines.append(f"{k},{predict[k]}\n")
    with open(out_file + '.txt', 'w') as f:
        f.writelines(txt_lines)
    with open(out_file + '.json', 'w') as f:
        json.dump(predict, f, indent=4)

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavdir', required=False, default = 'InputData/ttm/wavs',help='Directory with wav files')
    parser.add_argument('--outdir', required=False, default = 'TTM_eval_Output', help='Output directory for embeddings and results')
    parser.add_argument('--ckpt', required=False, default = 'model_ckpt/musiceval_ckpt/best_ckpt_28', help='Path to finetuned MLP checkpoint')
    parser.add_argument('--get_global', action='store_true', help='Extract global embeddings (1,dim)')
    parser.add_argument('--ssl_dim', type=int, default=768, help='SSL output dimension (default 768 for clamp3)')
    args = parser.parse_args()

    wavdir = args.wavdir
    outdir = args.outdir
    embed_dir = os.path.join(outdir, 'embeds')
    os.makedirs(outdir, exist_ok=True)

    # 1. extract embeddings
    print('Extracting embeddings...')
    extract_embeddings(wavdir, embed_dir, get_global=args.get_global)

    # 2. load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading checkpoint...')
    model = load_checkpoint(args.ckpt, device, ssl_out_dim=args.ssl_dim)

    # 3. run inference
    print('Running inference...')
    out_file = os.path.join(outdir, 'results')
    predictions = run_inference(embed_dir, model, device, out_file)
    print('Saved results to', out_file + '.txt and .json')

if __name__ == '__main__':
    main()
