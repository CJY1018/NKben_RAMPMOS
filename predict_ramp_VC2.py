import os
import pandas as pd
import fairseq
import torch, torchaudio

from model.ramp import MosPredictor


def predict_ramp(meta_df, checkpoint, datastore_path, device):
    max_k = 300
    topk = 8
    SSL_OUT_DIM = 768
    cp_path = 'model_ckpt/wav2vec_small.pt'
    my_checkpoint = checkpoint
    datastore_path = datastore_path
    
    if not os.path.exists(cp_path):
        os.system(f'wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P model_ckpt')
        os.system(f'wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P model_ckpt')
        
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    model = MosPredictor(ssl_model, SSL_OUT_DIM, max_k, datastore_path, topk=topk).to(device)
    model.eval()
    model.load_state_dict(torch.load(my_checkpoint, weights_only=False))
    
    predictions = {}

    print('start evaluate')
    
    for filepath in meta_df['infer_wav'].values.tolist():
        filename = os.path.basename(filepath)
        if filename.endswith('.wav'):  # Ensure we process only .wav files
            # Load the wav file
            wav, _ = torchaudio.load(filepath)
            inputs = wav.to(device)
            filenames = (filename.split('-')[0],)
            
            # Perform prediction
            with torch.no_grad():
                _, outputs = model(inputs, syslist=filenames)
            
            # Store predictions
            outputs = outputs.to('cpu').numpy()[0]
            predictions[filename] = outputs

    output_df = pd.DataFrame(list(predictions.items()), columns=['filename', 'ramp_mos'])
    
    avg_ramp = output_df['ramp_mos'].mean()
    avg_ramp = round(avg_ramp, 3)  # 保留三位小数
    
    return output_df, avg_ramp
    