import os
import pandas as pd
import fairseq
import torch, torchaudio

from model.ramp import MosPredictor


def predict_ramp(meta_df, checkpoint, datastore_path, device):
    max_k = 400
    topk = 1
    SSL_OUT_DIM = 768
    cp_path = '/mnt/storage/chenjunyang_space/RAMP_EVAL/NKben_RAMPMOS/model_ckpt/wav2vec_small.pt' #'model_ckpt/wav2vec_small.pt'
    my_checkpoint = checkpoint
    datastore_path = datastore_path
    
    if not os.path.exists(cp_path):
        os.system(f'wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P {cp_path}')
        os.system(f'wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P {cp_path}/')
        
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

    
    # 创建输出文件夹为output，输出文件名为ramp.csv
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ramp.csv")
    
    # generate ramp.csv
    output_df = None
    
    try:
        output_df = pd.DataFrame(list(predictions.items()), columns=['filename', 'ramp_mos'])
        output_df['filename'] = output_df['filename'].apply(lambda x: x.split('.')[0])  # Remove file extension
        output_df.to_csv(output_file, index=False, header=True, sep='\t')
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        return None
    
    avg_ramp = output_df['ramp_mos'].mean()
    
    return avg_ramp
    