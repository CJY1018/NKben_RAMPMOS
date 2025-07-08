import torch
import argparse

from process_meta_csv import prepare_wav_res_ref_text, save_meta_csv
from predict_wer import predict_wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], required=True, help='Language of the text (en or zh)')
    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    assert {'infer_wav', 'infer_text'}.issubset(meta_df.columns), "The meta CSV must contain 'infer_wav' and 'infer_text' columns."
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = args.meta_csv.split('meta_')[-1].split('.')[0]
    
    output_df, avg_wer = predict_wer(meta_df, args.lang, device=device)
    
    save_meta_csv(output_df, model_name, args.lang, 'wer')

    print(f"Average WER: {avg_wer}")


# export CUDA_VISIBLE_DEVICES=1
# python run_wer.py --meta_csv InputData/zh/meta_xtts.csv --lang zh