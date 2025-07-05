import torch
import argparse

from prepross_meta_csv import prepare_wav_res_ref_text
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
    
    if {'infer_wav', 'infer_text'}.issubset(meta_df.columns):
        avg_wer = predict_wer(meta_df, args.lang, device=device)
    else:
        print("No 'infer_wav' and 'infer_text' columns found in the meta CSV. Skipping WER prediction.")
        
    if {'infer_wav', 'prompt_wav'}.issubset(meta_df.columns):
        avg_sim = predict_sim(meta_df, model_name='wavlm_large', checkpoint=args.sim_checkpoint_path, wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=-1, wav2_end_sr=-1, wav2_cut_wav1=False, device=device)
    else:
        print("No 'infer_wav' and 'prompt_wav' columns found in the meta CSV. Skipping similarity prediction.")
        
    if 'infer_wav' in meta_df.columns:
        avg_ramp = predict_ramp(meta_df, checkpoint=args.ramp_checkpoint_path, datastore_path=args.datastore_path, device=device)
    else:
        print("No 'infer_wav' column found in the meta CSV. Skipping RAMP prediction.")
    
    print(f"Average WER: {avg_wer}")
    print(f"Average Similarity: {avg_sim}")
    print(f"Average RAMP: {avg_ramp}")


# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PYTHONPATH:fairseq
# export PYTHONPATH=$PYTHONPATH:thirdparty/UniSpeech/downstreams/speaker_verification
# python run_all.py --meta_csv new_example_samples/meta.csv --lang zh --sim_checkpoint_path wavlm_large_finetune.pth --ramp_checkpoint_path model_ckpt/ramp_ckpt --datastore_path datastore_profile