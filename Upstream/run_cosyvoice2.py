import os
import argparse
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current script directory: {current_script_dir}")
sys.path.append('Upstream/CosyVoice')
sys.path.append(os.path.join('CosyVoice', 'third_party', 'Matcha-TTS'))
import uuid
import torchaudio
import pandas as pd


from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav


cosyvoice = None

def run_tts(row, lang):
    global cosyvoice
    infer_text = row['infer_text']
    prompt_wav = row['prompt_wav']
    prompt_text = row['prompt_text']
    
    prompt_wav_path = f"InputData/{lang}/{prompt_wav}"
    save_wav_path = f"InputData/{lang}/wavs/{os.path.basename(prompt_wav)[:-4]}_cosyvoice2_{uuid.uuid4()}.wav"
    
    prompt_speech_16k = load_wav(prompt_wav_path, 16000)
    
    try:
        for _, j in enumerate(cosyvoice.inference_zero_shot(infer_text, prompt_text, prompt_speech_16k, stream=False)):
            torchaudio.save(save_wav_path, j['tts_speech'], cosyvoice.sample_rate)
        return os.path.basename(save_wav_path)  # Return the filename
    except Exception as e:
        print(f"Error processing text '{infer_text}': {e}")
        return None
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], help='Language to process: en/zh')
    args = parser.parse_args()
    
    cosyvoice = CosyVoice2('Upstream/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    
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
    output_meta_path = f"InputData/{args.lang}/meta_cosyvoice2.csv"
    
    # Reorder the column names
    meta_df = meta_df[["infer_wav", "infer_text", "prompt_wav", "prompt_text"]]
    
    meta_df.to_csv(output_meta_path, sep="|", index=False, header=False)
