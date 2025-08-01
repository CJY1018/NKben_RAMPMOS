import os
import argparse
import sys
sys.path.append('Upstream_VC/seed-vc')
import uuid
import torchaudio
import pandas as pd

from inference_v2 import main as vc_inference


def run_vc(row, lang):
    source_wav = row['source_wav']
    target_wav = row['target_wav']

    source_wav_path = f"InputData_VC/{lang}/{source_wav}"
    target_wav_path = f"InputData_VC/{lang}/{target_wav}"
    save_wav_dir = f"InputData_VC/{lang}/wavs"
    
    '''
    source is the path to the speech file to convert to reference voice
    target is the path to the speech file as voice reference
    output is the path to the output directory
    diffusion-steps is the number of diffusion steps to use, default is 25, use 30-50 for best quality, use 4-10 for fastest inference
    length-adjust is the length adjustment factor, default is 1.0, set <1.0 for speed-up speech, >1.0 for slow-down speech
    inference-cfg-rate has subtle difference in the output, default is 0.7
    '''
    args = argparse.Namespace(
        source=source_wav_path,
        target=target_wav_path,
        output=save_wav_dir,
        diffusion_steps=25, # recommended 30~50 for singing voice conversion
        length_adjust=1.0, # same as V1
        intelligibility_cfg_rate=0.7, # controls how clear the output linguistic content is, recommended 0.0~1.0
        similarity_cfg_rate=0.7, # controls how similar the output voice is to the reference voice, recommended 0.0~1.0
        convert_style=True, # whether to use AR model for accent & emotion conversion, set to false will only conduct timbre conversion similar to V1
        anonymization_only=False, # set to true will ignore reference audio but only anonymize source speech to an "average voice"
        top_p=0.9, # controls the diversity of the AR model output, recommended 0.5~1.0
        temperature=1.0, # controls the randomness of the AR model output, recommended 0.7~1.2
        repetition_penalty=1.0, # penalizes the repetition of the AR model output, recommended 1.0~1.5
        ar_checkpoint_path=None, # path to the checkpoint of the AR model, leave to blank to auto-download default model from huggingface
        cfm_checkpoint_path=None, # path to the checkpoint of the CFM model, leave to blank to auto-download default model from huggingface
        compile=False
    )
    
    filename = f"vc_v2_{os.path.splitext(os.path.basename(source_wav))[0]}_{os.path.splitext(os.path.basename(target_wav))[0]}_{args.length_adjust}_{args.diffusion_steps}_{args.similarity_cfg_rate}.wav"
    
    try:
        vc_inference(args)
        return filename
    except Exception as e:
        print(f"Error processing source '{source_wav}' and target '{target_wav}': {e}")
        return None
    


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
