import torch
import argparse

from process_meta_csv import prepare_wav_res_ref_text, save_meta_csv
from predict_utmos import predict_utmos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], required=True, help='Language of the text (en or zh)')
    parser.add_argument('--ckpt_path', type=str, default='MOS/UTMOS-demo/epoch=3-step=7459.ckpt', help='Path to the utmos model checkpoint')
    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    avg_wer = avg_sim = avg_ramp = None
    
    # 从meta_csv文件名中提取模型名称
    model_name = args.meta_csv.split('meta_')[-1].split('.')[0]
        
    if 'infer_wav' in meta_df.columns:
        output_df, avg_utmos = predict_utmos(meta_df, device=device, ckpt_path=args.ckpt_path)
        save_meta_csv(output_df, model_name, args.lang, 'utmos')
    else:
        print("No 'infer_wav' column found in the meta CSV. Skipping UTMOS prediction.")
        
    print(f"Average UTMOS({model_name}-{args.lang}): {avg_utmos}")


# ln -s MOS/UTMOS-demo/wav2vec_small.pt ./wav2vec_small.pt

# python run_all_utmos.py --meta_csv InputData/zh/meta_xtts.csv --lang zh --ckpt_path MOS/UTMOS-demo/epoch=3-step=7459.ckpt
# python run_all_utmos.py --meta_csv InputData/en/meta_cosyvoice2.csv --lang zh
# python run_all_utmos.py --meta_csv InputData/en/meta_cosyvoice2.csv --lang en