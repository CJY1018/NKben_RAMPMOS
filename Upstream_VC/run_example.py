import os
import argparse
# TODO: If the VC library is used, place it in the Upstream_VC directory and add it to the python environment
# import sys
# current_script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_script_dir, 'VC library'))
import uuid
import torchaudio
import pandas as pd

# TODO init vc_inference


def run_vc(row, lang):
    source_wav = row['source_wav']
    target_wav = row['target_wav']

    source_wav_path = f"InputData_VC/{lang}/{source_wav}"
    target_wav_path = f"InputData_VC/{lang}/{target_wav}"
    save_wav_dir = f"InputData_VC/{lang}/wavs"
    
    # TODO: Implement VC inference and save the output
    # save_wav_path = f"InputData/{lang}/wavs/VCMODEL_{os.path.splitext(os.path.basename(source_wav))[0]}_{os.path.splitext(os.path.basename(target_wav))[0]}.wav"

    # try:
        # wav = vc_inference(source_wav_path, target_wav_path)
        # torchaudio.save(save_wav_path, wav, sample_rate)
        # return os.path.basename(save_wav_path) # Return the filename
    # except Exception as e:
        # print(f"Error processing source '{source_wav}' and target '{target_wav}': {e}")
        # return None
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], help='Language to process: en/zh')
    args = parser.parse_args()
    
    # Load meta data
    meta_data_path = f"InputData_VC/{args.lang}/meta.csv"
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Meta data file not found: {meta_data_path}")
    
    meta_df = pd.read_csv(meta_data_path, sep="|", header=None, names=["source_wav", "target_wav"])
    
    wavs_dir = f"InputData_VC/{args.lang}/wavs"
    if not os.path.exists(wavs_dir):
        os.makedirs(wavs_dir)
    
    meta_df['infer_wav'] = meta_df.apply(run_vc, axis=1, args=(args.lang,))
    
    # Save updated meta data
    output_meta_path = f"InputData_VC/{args.lang}/meta_seed-vc.csv"
    
    # Reorder the column names
    meta_df = meta_df[["infer_wav", "source_wav", "target_wav"]]
    
    meta_df.to_csv(output_meta_path, sep="|", index=False, header=False)
