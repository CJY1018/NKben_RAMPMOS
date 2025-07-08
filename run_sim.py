import torch
import argparse

from process_meta_csv import prepare_wav_res_ref_text, save_meta_csv
from thirdparty.UniSpeech.downstreams.speaker_verification.verification_pair_list_v3 import predict_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], required=True, help='Language of the text (en or zh)')
    parser.add_argument('--sim_checkpoint_path', type=str, default='wavlm_large_finetune.pth', help='Path to the sim model checkpoint')
    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    assert {'infer_wav', 'prompt_wav'}.issubset(meta_df.columns), "The meta CSV must contain 'infer_wav' and 'prompt_wav' columns."
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = args.meta_csv.split('meta_')[-1].split('.')[0]
    
    output_df, avg_sim = predict_sim(meta_df, model_name='wavlm_large', checkpoint=args.sim_checkpoint_path, wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=-1, wav2_end_sr=-1, wav2_cut_wav1=False, device=device)
    
    save_meta_csv(output_df, model_name, args.lang, 'similarity')
    
    print(f"Average Similarity: {avg_sim}")


# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PYTHONPATH:thirdparty/UniSpeech/downstreams/speaker_verification
# python run_sim.py --meta_csv InputData/zh/meta_xtts.csv --lang zh --sim_checkpoint_path wavlm_large_finetune.pth