import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_metric_dict(output_dir, model_list):
    metric_dict = {}  # 字典结构: {模型名: {语言: {指标: 值}}}

    for file in os.listdir(output_dir):
        if file.endswith('.csv') and any([item in file for item in model_list]):
            file_path = os.path.join(output_dir, file)
            
            try:
                # 解析文件名 (格式: 指标_模型名_语言.csv)
                parts = file.split('.')[0].split('_')
                metric, model_name, lang = parts[0], parts[1], parts[2]
                
                # 读取CSV文件
                df = pd.read_csv(file_path, sep='\t', header=0)
                
                # 初始化嵌套字典
                if model_name not in metric_dict:
                    metric_dict[model_name] = {}
                if lang not in metric_dict[model_name]:
                    metric_dict[model_name][lang] = {}
                    
                if metric == 'similarity':
                    avg_sim = df['similarity'].mean() # 宏平均 (Macro-average)
                    avg_sim = round(avg_sim, 3)  # 保留三位小数
                    metric_dict[model_name][lang]['similarity'] = avg_sim
                elif metric == 'ramp':
                    avg_ramp = df['ramp_mos'].mean() # 宏平均 (Macro-average)
                    avg_ramp = round(avg_ramp, 3)
                    metric_dict[model_name][lang]['ramp'] = avg_ramp
                elif metric == 'sslmos':
                    avg_ramp = df['sslmos'].mean()
                    avg_ramp = round(avg_ramp, 3)
                    metric_dict[model_name][lang]['sslmos'] = avg_ramp
                elif metric == 'audiobox':
                    avg_ramp = df['audiobox'].mean()
                    avg_ramp = round(avg_ramp, 3)
                    metric_dict[model_name][lang]['audiobox'] = avg_ramp
                elif metric == 'utmos':
                    avg_ramp = df['utmos'].mean()
                    avg_ramp = round(avg_ramp, 3)
                    metric_dict[model_name][lang]['utmos'] = avg_ramp
                elif metric == 'gtmos':
                    avg_mos = df['gt_mos'].mean() # 宏平均 (Macro-average)
                    avg_mos = round(avg_mos, 3)  # 保留三位小数
                    metric_dict[model_name][lang]['gt_mos'] = avg_mos
                    
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")

    return metric_dict


if __name__ == "__main__":
    output_dir = 'output'  # 输出目录
    model_list = ["seed-vc"] # 定义需要计算的模型
    metric_dict = get_metric_dict(output_dir, model_list)
    
    print("Metric Dictionary:", metric_dict)
    
    # 保存metric_dict为JSON文件
    metric_list = [{model: metric_dict[model]} for model in metric_dict ]
    with open(os.path.join(output_dir, 'metric_vc.json'), 'w', encoding='utf-8') as f:
        json.dump(metric_list, f, ensure_ascii=False, indent=4)
        