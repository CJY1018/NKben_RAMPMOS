import os
import pandas as pd
from tqdm import tqdm
from verification import verification


def predict_sim(meta_df, model_name, checkpoint, wav1_start_sr, wav2_start_sr, wav1_end_sr, wav2_end_sr, wav2_cut_wav1, device):
    params = meta_df[['infer_wav', 'prompt_wav']].values.tolist()
    
    # 判定两列均不为空
    assert all(pd.notna(meta_df['infer_wav'])) and all(pd.notna(meta_df['prompt_wav'])), "infer_wav and prompt_wav columns must not contain NaN values"

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
        
    # 创建输出文件夹为output，输出文件名为sim.csv
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sim.csv")

    output_df = None
    
    try:
        output_df = pd.DataFrame(sim_lines, columns=['infer_wav', 'wav1_start_sr', 'wav1_end_sr', 'prompt_wav', 'wav2_start_sr', 'wav2_end_sr', 'similarity'])
        output_df.to_csv(output_file, index=False, header=True, sep='\t')
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        return None
    
    avg_score = round(output_df['similarity'].mean(), 3)
    
    return avg_score
