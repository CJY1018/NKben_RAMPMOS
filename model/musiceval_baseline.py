"""
MusicEval中的MosPredictor模型，使用CLAMP3提取的音频和文本嵌入作为输入特征
"""
import os
import torch
import torch.nn as nn
import numpy as np
import random
random.seed(1984)

def load_clamp3_from_filename(wav_name, audio_embed_dir, text_embed_dir, device: torch.device = None):
    """
    input: 'path/to/S001_P001.wav'
    return: (audio_embeds_tensor, text_embeds_tensor)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 支持 full path 或 basename
    base = os.path.basename(wav_name)
    name = base.split('.')[0]
    parts = name.split('_')
    s_id = parts[0] if len(parts) > 0 else name # 'S001'
    p_id = parts[1] if len(parts) > 1 else name # 'P001'

    audio_embed_file = os.path.join(audio_embed_dir, name + '.npy')
    text_embed_file = os.path.join(text_embed_dir, p_id + '.npy')

    # 加载 numpy 文件
    audio_embed = np.load(audio_embed_file)
    text_embed = np.load(text_embed_file)

    audio_embed = torch.tensor(audio_embed).squeeze(1).to(device)
    text_embed = torch.tensor(text_embed).squeeze(1).to(device)
    return audio_embed, text_embed

class MosPredictor(nn.Module):
    def __init__(self, up_out_dim):
        super(MosPredictor, self).__init__()
        self.upstream_feat_dim = up_out_dim
        self.overall_mlp_layer1 = nn.Linear(in_features = self.upstream_feat_dim, out_features = 256)
        self.overall_mlp_layer2 = nn.Linear(in_features = 256, out_features = 64)
        self.overall_mlp_layer3 = nn.Linear(in_features = 64, out_features = 1)
        self.adherence_mlp_layer1 = nn.Linear(in_features = self.upstream_feat_dim*2, out_features = 256)
        self.adherence_mlp_layer2 = nn.Linear(in_features = 256, out_features = 64)
        self.adherence_mlp_layer3 = nn.Linear(in_features = 64, out_features = 1)


    def forward(self, audio_embeds, text_embeds):
        """
        input:(bs, 768)
        return:(bs, 1)
        """ 
        combine_embed=torch.cat((audio_embeds,text_embeds),dim=1) # bs*(768*2)        
        hidden1 = self.overall_mlp_layer1(audio_embeds)
        hidden1_2 = self.overall_mlp_layer2(hidden1)
        out1 = self.overall_mlp_layer3(hidden1_2)

        hidden2 = self.adherence_mlp_layer1(combine_embed)
        hidden2_2 = self.adherence_mlp_layer2(hidden2)
        out2 = self.adherence_mlp_layer3(hidden2_2)
        return out1,out2