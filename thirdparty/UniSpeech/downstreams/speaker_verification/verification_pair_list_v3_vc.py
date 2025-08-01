import os
import sys
sys.path.append('thirdparty/UniSpeech/downstreams/speaker_verification')
import pandas as pd
from tqdm import tqdm
from verification import verification


def predict_sim(meta_df, model_name, checkpoint, wav1_start_sr, wav2_start_sr, wav1_end_sr, wav2_end_sr, wav2_cut_wav1, device):
    params = meta_df[['infer_wav', 'target_wav']].values.tolist()
    
    # 判定两列均不为空
    assert all(pd.notna(meta_df['infer_wav'])) and all(pd.notna(meta_df['target_wav'])), "infer_wav and target_wav columns must not contain NaN values"

    model = None
    sim_lines = []
    
    for t1, t2 in tqdm(params):
        t1_path = t1.strip()
        t2_path = t2.strip()
        if not os.path.exists(t1_path) or not os.path.exists(t2_path):
            continue
        try:
            sim, model = verification(model_name, t1_path, t2_path, use_gpu=True, checkpoint=checkpoint, wav1_start_sr=wav1_start_sr, wav2_start_sr=wav2_start_sr, wav1_end_sr=wav1_end_sr, wav2_end_sr=wav2_end_sr, model=model, wav2_cut_wav1=wav2_cut_wav1, device=device)
        except Exception as e:
            print(str(e))
            continue

        if sim is None:
            continue
        
        sim_lines.append((t1_path, wav1_start_sr, wav1_end_sr, t2_path, wav2_start_sr, wav2_end_sr, sim.cpu().item()))
        
    output_df = pd.DataFrame(sim_lines, columns=['infer_wav', 'wav1_start_sr', 'wav1_end_sr', 'target_wav', 'wav2_start_sr', 'wav2_end_sr', 'similarity'])
    
    avg_score = round(output_df['similarity'].mean(), 3)
    
    return output_df, avg_score
