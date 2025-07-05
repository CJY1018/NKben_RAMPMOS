import os
import pandas as pd
from tqdm import tqdm
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel


def load_en_model(device):
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def load_zh_model():
    model = AutoModel(model="paraformer-zh")
    return model

def process_one(hypo, truth, lang):
    raw_truth = truth
    raw_hypo = hypo
    
    punctuation_all = punctuation + string.punctuation

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)

def predict_wer(meta_df, lang, device):
    if lang == "en":
        processor, model = load_en_model(device)
    elif lang == "zh":
        model = load_zh_model()
        
    params = meta_df[['infer_wav', 'infer_text']].values.tolist()
    
    wer_lines = []
    
    for infer_wav, infer_text in tqdm(params):
        if lang == "en":
            wav, sr = sf.read(infer_wav)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif lang == "zh":
            res = model.generate(input=infer_wav,
                    batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, 'zh-cn')

        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, infer_text, lang)
        wer_lines.append((infer_wav, wer, raw_truth, raw_hypo, inse, dele, subs))
        
    # 创建输出文件夹为output，输出文件名为wer.csv
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "wer.csv")
    
    output_df = None
    
    try:
        output_df = pd.DataFrame(wer_lines, columns=['infer_wav', 'wer', 'infer_text', 'asr_text', 'wer_ins', 'wer_del', 'wer_sub'])
        output_df.to_csv(output_file, index=False, header=True, sep='\t')
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        return None
    
    avg_wer = round(output_df['wer'].mean() * 100, 3)
    
    return avg_wer