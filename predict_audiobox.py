import sys
sys.path.append("MOS/audiobox-aesthetics")
import os
import torchaudio
import pandas as pd

from audiobox_aesthetics.infer import initialize_predictor


def predict_audiobox(meta_df, metric='PQ'):
    
    predictor = initialize_predictor()
    
    predictions = {}
    
    for filepath in meta_df['infer_wav'].values.tolist():
        filename = os.path.basename(filepath)
        if filename.endswith('.wav'):  # Ensure we process only .wav files
            # Load the wav file
            wav, sr = torchaudio.load(filepath)
            score = predictor.forward([{"path":wav, "sample_rate": sr}])
            # 'CE', 'CU', 'PC', 'PQ'
            predictions[filename] = score[0][metric]

    output_df = pd.DataFrame(list(predictions.items()), columns=['filename', 'audiobox'])
    
    avg_ramp = output_df['audiobox'].mean()
    avg_ramp = round(avg_ramp, 3)  # 保留三位小数
    
    return output_df, avg_ramp


    