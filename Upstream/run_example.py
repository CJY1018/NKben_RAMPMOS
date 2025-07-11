import os
import argparse
# TODO: If the TTS library is used, place it in the Upstream directory and add it to the python environment
# import sys
# current_script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_script_dir, 'TTS library'))
import uuid
import torch
import pandas as pd


tts = None

def run_tts(row, lang):
    global tts
    infer_text = row['infer_text']
    prompt_wav = row['prompt_wav']
    prompt_text = row['prompt_text']
    
    prompt_wav_path = f"InputData/{lang}/{prompt_wav}"
    save_wav_path = f"InputData/{lang}/wavs/{os.path.basename(prompt_wav)[:-4]}_xtts_{uuid.uuid4()}.wav"
    
    # Run TTS
    try:
        # TODO: Implement TTS inference and save the output
        # wav = tts.inference(infer_text, prompt_wav_path, prompt_text, lang)
        # torchaudio.save(save_wav_path, wav, sample_rate)
        
        return os.path.basename(save_wav_path) # Return the filename
    except Exception as e:
        print(f"Error processing text '{infer_text}': {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], help='Language to process: en/zh')
    args = parser.parse_args()
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: Init TTS
    # tts = TTS().to(device)
    
    
    # Load meta data
    meta_data_path = f"InputData/{args.lang}/meta.csv"
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Meta data file not found: {meta_data_path}")
    
    meta_df = pd.read_csv(meta_data_path, sep="|", header=None, names=["infer_text", "prompt_wav", "prompt_text"])
    
    wavs_dir = f"InputData/{args.lang}/wavs"
    if not os.path.exists(wavs_dir):
        os.makedirs(wavs_dir)
    
    meta_df['infer_wav'] = meta_df.apply(run_tts, axis=1, args=(args.lang,))
    
    # Save updated meta data
    output_meta_path = f"InputData/{args.lang}/meta_xtts.csv"
    
    # Reorder the column names
    meta_df = meta_df[["infer_wav", "infer_text", "prompt_wav", "prompt_text"]]
    
    meta_df.to_csv(output_meta_path, sep="|", index=False, header=False)
            