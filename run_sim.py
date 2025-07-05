import torch
import argparse

from prepross_meta_csv import prepare_wav_res_ref_text
from thirdparty.UniSpeech.downstreams.speaker_verification.verification_pair_list_v3 import predict_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--sim_checkpoint_path', type=str, default='wavlm_large_finetune.pth', help='Path to the sim model checkpoint')
    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    assert {'infer_wav', 'prompt_wav'}.issubset(meta_df.columns), "The meta CSV must contain 'infer_wav' and 'prompt_wav' columns."
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    avg_sim = predict_sim(meta_df, model_name='wavlm_large', checkpoint=args.sim_checkpoint_path, wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=-1, wav2_end_sr=-1, wav2_cut_wav1=False, device=device)
    
    print(f"Average Similarity: {avg_sim}")


# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PYTHONPATH:thirdparty/UniSpeech/downstreams/speaker_verification
# python run_sim.py --meta_csv new_example_samples/meta.csv --sim_checkpoint_path wavlm_large_finetune.pth