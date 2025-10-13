import os
import pandas as pd
import torchaudio
import torchaudio
import sys
sys.path.append('MOS/UTMOS-demo')
from score import Score


def predict_utmos(meta_df, device, ckpt_path="MOS/UTMOS-demo/epoch=3-step=7459.ckpt"):
    predictions = {}
    
    for filepath in meta_df['infer_wav'].values.tolist():
        filename = os.path.basename(filepath)
        if filename.endswith('.wav'):  # Ensure we process only .wav files
            # Load the wav file
            wav, sr = torchaudio.load(filepath)
            scorer = Score(ckpt_path=ckpt_path, input_sample_rate=sr, device=device)
            score = scorer.score(wav.to(device))
            predictions[filename] = score[0]

    output_df = pd.DataFrame(list(predictions.items()), columns=['filename', 'utmos'])
    
    avg_ramp = output_df['utmos'].mean()
    avg_ramp = round(avg_ramp, 3)  # 保留三位小数
    
    return output_df, avg_ramp
    