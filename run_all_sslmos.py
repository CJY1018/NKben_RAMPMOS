import os
import torch
import argparse

from process_meta_csv import prepare_wav_res_ref_text, save_meta_csv
from predict_sslmos import predict_sslmos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], required=True, help='Language of the text (en or zh)')
    parser.add_argument('--fairseq_base_model', type=str, default='MOS/mos-finetune-ssl/fairseq/wav2vec_small.pt', help='Path to the wav2vec_small model checkpoint')
    parser.add_argument('--finetuned_checkpoint', type=str, default='MOS/mos-finetune-ssl/pretrained/ckpt_w2vsmall', help='Path to the ckpt_w2vsmall model checkpoint')

    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    wavdir = os.path.join(os.path.dirname(args.meta_csv) , 'wavs')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    avg_wer = avg_sim = avg_ramp = None
    
    # 从meta_csv文件名中提取模型名称
    model_name = args.meta_csv.split('meta_')[-1].split('.')[0]
        
    if 'infer_wav' in meta_df.columns:
        output_df, avg_sslmos = predict_sslmos(meta_df, wavdir=wavdir, device=device, fairseq_base_model=args.fairseq_base_model, finetuned_checkpoint=args.finetuned_checkpoint)
        save_meta_csv(output_df, model_name, args.lang, 'sslmos')
    else:
        print("No 'infer_wav' column found in the meta CSV. Skipping UTMOS prediction.")
        
    
    print(f"Average SSLMOS({model_name}-{args.lang}): {avg_sslmos}")


# python run_all_sslmos.py --meta_csv InputData/zh/meta_xtts.csv --lang zh --fairseq_base_model MOS/mos-finetune-ssl/fairseq/wav2vec_small.pt --finetuned_checkpoint MOS/mos-finetune-ssl/pretrained/ckpt_w2vsmall
# python run_all_sslmos.py --meta_csv InputData/en/meta_cosyvoice2.csv --lang zh
# python run_all_sslmos.py --meta_csv InputData/en/meta_cosyvoice2.csv --lang en