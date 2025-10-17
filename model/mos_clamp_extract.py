"""
CLAP替换为clamp，提取并训练下游输出头，
"""
import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import random
random.seed(1984)
from utils import *

AUDIO_CLAMP_PATH = "/home/liucheng/project/mos-finetune-ssl/data/musiceval_clamp3_embed/audio_embed/"
TEXT_CLAMP_PATH = "/home/liucheng/project/mos-finetune-ssl/data/musiceval_clamp3_embed/text_embed/"

def load_clamp3_from_filename(filenames):
    audio_embeds=[]
    text_embeds=[]

    for wav_name in filenames:
        # wav_name:S001_P001.wav
        wav_name=wav_name.split(".")[0]
        # wav_name:S001_P001
        s_id = wav_name.split("_")[0]
        p_id = wav_name.split("_")[1]

        if s_id in SPECIAL_S_IDS:   # demo_prompt_info
            text_embed_file = TEXT_CLAMP_PATH + wav_name + ".npy"
        else:   # prompt_info
            text_embed_file = TEXT_CLAMP_PATH + p_id + ".npy"

        audio_embed_file = AUDIO_CLAMP_PATH + wav_name + ".npy"

        audio_embed = np.load(file=audio_embed_file)
        text_embed = np.load(file=text_embed_file)
        # print("【ok", wav_name)
        text_embeds.append(text_embed)
        audio_embeds.append(audio_embed)
    audio_embeds=torch.tensor(audio_embeds).squeeze(1).to(device)
    text_embeds=torch.tensor(text_embeds).squeeze(1).to(device)
    # print("【audio_embed.shape", audio_embeds.shape) # (64,768)
    return audio_embeds, text_embeds

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
        input:(64,768)
        return:(64,1)
        """ 
        combine_embed=torch.cat((audio_embeds,text_embeds),dim=1) # bs*(768*2)        
        hidden1 = self.overall_mlp_layer1(audio_embeds)
        hidden1_2 = self.overall_mlp_layer2(hidden1)
        out1 = self.overall_mlp_layer3(hidden1_2)

        hidden2 = self.adherence_mlp_layer1(combine_embed)
        hidden2_2 = self.adherence_mlp_layer2(hidden2)
        out2 = self.adherence_mlp_layer3(hidden2_2)
        return out1,out2