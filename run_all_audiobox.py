import argparse

from process_meta_csv import prepare_wav_res_ref_text, save_meta_csv
from predict_audiobox import predict_audiobox


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to the meta CSV file')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], required=True, help='Language of the text (en or zh)')
    parser.add_argument('--metric', type=str, default='PQ', help="Choose one from the 'CE', 'CU', 'PC', 'PQ'")
    args = parser.parse_args()

    meta_df = prepare_wav_res_ref_text(args.meta_csv)
    
    avg_wer = avg_sim = avg_ramp = None
    
    # 从meta_csv文件名中提取模型名称
    model_name = args.meta_csv.split('meta_')[-1].split('.')[0]
        
    if 'infer_wav' in meta_df.columns:
        output_df, avg_audiobox = predict_audiobox(meta_df, metric=args.metric)
        save_meta_csv(output_df, model_name, args.lang, 'audiobox')
    else:
        print("No 'infer_wav' column found in the meta CSV. Skipping AUDIOBOX prediction.")
        
    print(f"Average AUDIOBOX({model_name}-{args.lang}-{args.metric}): {avg_audiobox}")


# python run_all_audiobox.py --meta_csv InputData/zh/meta_xtts.csv --lang zh --metric PQ
# python run_all_audiobox.py --meta_csv InputData/en/meta_cosyvoice2.csv --lang en
# python run_all_audiobox.py --meta_csv InputData/en/meta_cosyvoice2.csv --lang en