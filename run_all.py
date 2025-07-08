import torch
import argparse

from process_meta_csv import prepare_wav_res_ref_text, save_meta_csv
from predict_wer import predict_wer
from thirdparty.UniSpeech.downstreams.speaker_verification.verification_pair_list_v3 import predict_sim
from predict_ramp2 import predict_ramp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], required=True, help='Language of the text (en or zh)')
    parser.add_argument('--sim_checkpoint_path', type=str, default='wavlm_large_finetune.pth', help='Path to the sim model checkpoint')
    parser.add_argument('--ramp_checkpoint_path', type=str, default='model_ckpt/ramp_ckpt', help='Path to the RAMP model checkpoint')
    parser.add_argument('--datastore_path', type=str, default='datastore_profile', help='Path to the datastore for RAMP')
    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    avg_wer = avg_sim = avg_ramp = None
    
    # 从meta_csv文件名中提取模型名称
    model_name = args.meta_csv.split('meta_')[-1].split('.')[0]
    
    # 计算WER
    if {'infer_wav', 'infer_text'}.issubset(meta_df.columns):
        output_df, avg_wer = predict_wer(meta_df, args.lang, device=device)
        save_meta_csv(output_df, model_name, args.lang, 'wer')
    else:
        print("No 'infer_wav' and 'infer_text' columns found in the meta CSV. Skipping WER prediction.")
    
    # 计算相似度
    if {'infer_wav', 'prompt_wav'}.issubset(meta_df.columns):
        output_df, avg_sim = predict_sim(meta_df, model_name='wavlm_large', checkpoint=args.sim_checkpoint_path, wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=-1, wav2_end_sr=-1, wav2_cut_wav1=False, device=device)
        save_meta_csv(output_df, model_name, args.lang, 'similarity')
    else:
        print("No 'infer_wav' and 'prompt_wav' columns found in the meta CSV. Skipping similarity prediction.")
    
    # 计算RAMP
    if 'infer_wav' in meta_df.columns:
        output_df, avg_ramp = predict_ramp(meta_df, checkpoint=args.ramp_checkpoint_path, datastore_path=args.datastore_path, device=device)
        save_meta_csv(output_df, model_name, args.lang, 'ramp')
    else:
        print("No 'infer_wav' column found in the meta CSV. Skipping RAMP prediction.")
    
    print("*" * 50)
    print(f"Average WER({model_name}-{args.lang}): {avg_wer}")
    print(f"Average Similarity({model_name}-{args.lang}): {avg_sim}")
    print(f"Average RAMP({model_name}-{args.lang}): {avg_ramp}")
    print("*" * 50)


# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PYTHONPATH:fairseq
# export PYTHONPATH=$PYTHONPATH:thirdparty/UniSpeech/downstreams/speaker_verification
# python run_all.py --meta_csv InputData/zh/meta_xtts.csv --lang zh --sim_checkpoint_path wavlm_large_finetune.pth --ramp_checkpoint_path model_ckpt/ramp_ckpt --datastore_path datastore_profile
# python run_all.py --meta_csv InputData/zh/meta_xtts.csv --lang zh
# python run_all.py --meta_csv InputData/en/meta_cosyvoice2.csv --lang en