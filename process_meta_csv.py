import os
import pandas as pd

# 读取并处理meta_csv文件，生成包含完整路径的infer_wav和prompt_wav列
def prepare_wav_res_ref_text(meta_csv):
    meta_df = pd.read_csv(meta_csv, header=None, sep='|')
    meta_df.columns = ['infer_wav', 'infer_text', 'prompt_wav', 'prompt_text'][:len(meta_df.columns)]

    # 修改infer_wav文件名，使其包含完整路径
    meta_df['infer_wav'] = meta_df['infer_wav'].apply(lambda x: os.path.join(os.path.dirname(meta_csv), 'wavs', x))

    # 如果prompt_wav列存在，则也修改其文件名
    if 'prompt_wav' in meta_df.columns:
        meta_df['prompt_wav'] = meta_df['prompt_wav'].apply(lambda x: os.path.join(os.path.dirname(meta_csv), x))
        
    return meta_df

# 保存结果到CSV文件
def save_meta_csv(output_df, model_name, lang, metric):
    # 创建输出文件夹为output，输出文件名为wer.csv
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{metric}_{model_name}_{lang}.csv")
    
    try:
        # 保存结果到CSV文件
        output_df.to_csv(output_file, index=False, header=True, sep='\t')
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        return None