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
        meta_df['prompt_wav'] = meta_df['prompt_wav'].apply(lambda x: os.path.join(os.path.dirname(meta_csv), 'prompt_wavs', x))
        
    return meta_df