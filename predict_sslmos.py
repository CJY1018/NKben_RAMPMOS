import os
import torch
import pandas as pd
import torch.nn as nn
import sys
sys.path.append('MOS/mos-finetune-ssl/fairseq-0.10.2')
sys.path.append('MOS/mos-finetune-ssl')
import fairseq
from torch.utils.data import DataLoader
from mos_fairseq import MosPredictor, MyDataset
import datetime
import time

def unixnow():
    return str(int(time.mktime(datetime.datetime.now().timetuple())))


def systemID(uttID):
    return uttID.split('-')[0]


def predict_sslmos(meta_df, wavdir, device, fairseq_base_model='MOS/mos-finetune-ssl/fairseq/wav2vec_small.pt', finetuned_checkpoint='MOS/mos-finetune-ssl/pretrained/ckpt_w2vsmall'):
    cp_path = fairseq_base_model
    my_checkpoint = finetuned_checkpoint
    device = device
    
    
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    
    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()
        
        
    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))

    wavfnames = [os.path.basename(x) for x in meta_df['infer_wav'].values.tolist()]
    wavlist = 'tmp_' + unixnow() + '.txt'
    wavlistf = open(wavlist, 'w')
    for w in wavfnames:
        wavlistf.write(w + ',3.0\n')
    wavlistf.close()        
    
    
    print('Loading data')
    validset = MyDataset(wavdir, wavlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.L1Loss()
    print('Starting prediction')

    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        output = outputs.cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1
        
    
    output_df = pd.DataFrame(list(predictions.items()), columns=['filename', 'sslmos'])
    
    avg_ramp = output_df['sslmos'].mean()
    avg_ramp = round(avg_ramp, 3)  # 保留三位小数
    
    return output_df, avg_ramp

    