import torch
import argparse

from prepross_meta_csv import prepare_wav_res_ref_text
from predict_ramp2 import predict_ramp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--ramp_checkpoint_path', type=str, default='model_ckpt/ramp_ckpt', help='Path to the RAMP model checkpoint')
    parser.add_argument('--datastore_path', type=str, default='datastore_profile', help='Path to the datastore for RAMP')
    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    assert 'infer_wav' in meta_df.columns, "The meta CSV must contain 'infer_wav' column."
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    avg_ramp = predict_ramp(meta_df, checkpoint=args.ramp_checkpoint_path, datastore_path=args.datastore_path, device=device)
    
    print(f"Average RAMP: {avg_ramp}")


# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PYTHONPATH:fairseq
# python run_ramp.py --meta_csv new_example_samples/meta.csv --ramp_checkpoint_path model_ckpt/ramp_ckpt --datastore_path datastore_profile