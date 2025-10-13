import torch
import argparse

from process_meta_csv_vc import prepare_wav_res_ref_text, save_meta_csv
from thirdparty.UniSpeech.downstreams.speaker_verification.verification_pair_list_v3_vc import predict_sim
from predict_ramp_VC2 import predict_ramp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], required=True, help='Language of the text (en or zh)')
    parser.add_argument('--sim_checkpoint_path', type=str, default='wavlm_large_finetune.pth', help='Path to the sim model checkpoint')
    parser.add_argument('--ramp_checkpoint_path', type=str, default='model_ckpt/ramp_ckpt_vc', help='Path to the RAMP model checkpoint')
    parser.add_argument('--datastore_path', type=str, default='datastore_profile', help='Path to the datastore for RAMP')
    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    avg_wer = avg_sim = avg_ramp = None
    
    # 从meta_csv文件名中提取模型名称
    model_name = args.meta_csv.split('meta_')[-1].split('.')[0]
    
    # 计算相似度
    if {'infer_wav', 'target_wav'}.issubset(meta_df.columns):
        output_df, avg_sim = predict_sim(meta_df, model_name='wavlm_large', checkpoint=args.sim_checkpoint_path, wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=-1, wav2_end_sr=-1, wav2_cut_wav1=False, device=device)
        save_meta_csv(output_df, model_name, args.lang, 'similarity')
    else:
        print("No 'infer_wav' and 'target_wav' columns found in the meta CSV. Skipping similarity prediction.")
    
    # 计算RAMP
    if 'infer_wav' in meta_df.columns:
        output_df, avg_ramp = predict_ramp(meta_df, checkpoint=args.ramp_checkpoint_path, datastore_path=args.datastore_path, device=device)
        save_meta_csv(output_df, model_name, args.lang, 'ramp')
    else:
        print("No 'infer_wav' column found in the meta CSV. Skipping RAMP prediction.")
    
    print("*" * 50)
    print(f"Average Similarity({model_name}-{args.lang}): {avg_sim}")
    print(f"Average RAMP({model_name}-{args.lang}): {avg_ramp}")
    print("*" * 50)
