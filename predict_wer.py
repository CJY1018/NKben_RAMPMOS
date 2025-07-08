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
        
    output_df = pd.DataFrame(wer_lines, columns=['infer_wav', 'wer', 'infer_text', 'asr_text', 'wer_ins', 'wer_del', 'wer_sub'])
    
    word_counts = output_df['infer_text'].str.split().apply(len)
    avg_wer = (output_df['wer'] * word_counts).sum() / word_counts.sum()  # 微平均 (Micro‑average)／加权平均，不影响中文计算
    avg_wer = round(avg_wer * 100, 3)  # 转换为百分比并保留三位小数
    
    # avg_wer = round(output_df['wer'].mean() * 100, 3) # 宏平均 (Macro-average)
    
    return output_df, avg_wer