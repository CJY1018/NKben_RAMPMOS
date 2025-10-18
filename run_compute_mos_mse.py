import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 复用 run_plot.py 的思路，读取 output 下各模型的聚合指标
# 文件命名: {metric}_{model}_{lang}.csv，metric in [ramp, sslmos, audiobox, utmos, gtmos]
# 列: ramp_mos/sslmos/audiobox/utmos/gt_mos

METRICS = ["ramp", "sslmos", "audiobox", "utmos", "gtmos"]
SCORE_COLS = {
    "ramp": "ramp_mos",
    "sslmos": "sslmos",
    "audiobox": "audiobox",
    "utmos": "utmos",
    "gtmos": "gt_mos",
}


def get_metric_dict(output_dir: str, model_list):
    """返回 {model: {lang: {metric: avg}}}，均值保留3位小数
    同时返回 MSE 字典: {model: {lang: {metric: mse}}}
    """
    metric_dict = {}
    mse_dict = {}
    # 先加载所有数据，以便后续按 filename 匹配计算 MSE
    all_data = {}

    for file in os.listdir(output_dir):
        if not file.endswith('.csv'):
            continue
        # 仅处理指定模型且在 METRICS 范围内的文件
        name = file.split('.')[0]
        parts = name.split('_')
        if len(parts) < 3:
            continue
        metric, model_name, lang = parts[0], parts[1], parts[2]
        if metric not in METRICS:
            continue
        if model_list and model_name not in model_list:
            continue

        file_path = os.path.join(output_dir, file)
        try:
            df = pd.read_csv(file_path, sep='\t', header=0)
            score_col = SCORE_COLS[metric]
            if score_col not in df.columns:
                # 容错：有些 CSV 可能分隔符为逗号
                df = pd.read_csv(file_path)
                if score_col not in df.columns:
                    print(f"跳过 {file}，找不到列 {score_col}")
                    continue

            # 确保有 filename 列用于匹配
            if 'filename' not in df.columns:
                print(f"警告: {file} 没有 filename 列")
                continue
            
            scores = df[score_col].values.copy()
            
            # Audiobox 归一化: 1-10 -> 1-5 (除以2)
            if metric == 'audiobox':
                scores = scores / 2.0
            
            avg = round(float(scores.mean()), 3)
            metric_dict.setdefault(model_name, {}).setdefault(lang, {})[metric] = avg
            
            # 保存数据用于后续 MSE 计算，重命名列以便后续匹配
            key = (model_name, lang, metric)
            temp_df = df[['filename']].copy()
            temp_df['score'] = scores  # 统一列名为 'score'
            all_data[key] = temp_df
                
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")

    # 计算 MSE: 将各指标与 gtmos 按 filename 匹配
    for model_name in metric_dict:
        for lang in metric_dict[model_name]:
            gt_key = (model_name, lang, 'gtmos')
            if gt_key not in all_data:
                continue
            
            gt_df = all_data[gt_key].rename(columns={'score': 'gt_score'})
            
            for metric in ['ramp', 'sslmos', 'audiobox', 'utmos']:
                metric_key = (model_name, lang, metric)
                if metric_key not in all_data:
                    continue
                
                metric_df = all_data[metric_key].rename(columns={'score': 'pred_score'})
                
                # 按 filename 合并
                merged = pd.merge(metric_df, gt_df, on='filename', how='inner')
                
                if len(merged) == 0:
                    continue
                
                mse = float(((merged['pred_score'] - merged['gt_score']) ** 2).mean())
                mse = round(mse, 4)
                mse_dict.setdefault(model_name, {}).setdefault(lang, {})[metric] = mse

    return metric_dict, mse_dict


def plot_bars_for_language(metric_dict, language: str, output_dir: str, model_colors=None):
    """
    对给定语言绘制每个模型一张图：
    - X: 四个指标 ramp/sslmos/audiobox/utmos
    - Y: 对应分数
    - gtmos: 以一条水平虚线显示（同图中每个模型线各自按其 gtmos 值画）
    生成: output/bar_{language}_{model}.png
    """
    metrics_for_bar = ["ramp", "sslmos", "audiobox", "utmos"]

    models = [m for m in metric_dict.keys() if language in metric_dict[m]]
    if not models:
        print(f"语言 {language} 无可用数据")
        return []

    if model_colors is None:
        default_colors = ['#ffaf00', '#f46920', '#1f77b4', '#2ca02c']
        model_colors = {m: default_colors[i % len(default_colors)] for i, m in enumerate(models)}

    saved = []

    # 每个模型单独一张图，突出对比 ramp 与其它 MOS
    for model in models:
        data = metric_dict[model][language]
        scores = [data.get(m, np.nan) for m in metrics_for_bar]
        gt_val = data.get('gtmos', None)

        plt.figure(figsize=(7, 5))
        x = np.arange(len(metrics_for_bar))
        color = model_colors.get(model, '#1f77b4')

        bars = plt.bar(x, scores, color=color, alpha=0.85)

        # 在柱顶部标注数值
        for b, s in zip(bars, scores):
            if s is not None and not (isinstance(s, float) and np.isnan(s)):
                plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{s:.2f}",
                         ha='center', va='bottom', fontsize=10)

        # gtmos 虚线
        if gt_val is not None:
            # audiobox 的 GT 在雷达图里是 2 倍，这里维持原始刻度对齐 score
            plt.axhline(y=gt_val, linestyle='--', color='red', linewidth=2, label='GT MOS')
            plt.text(len(metrics_for_bar)-0.4, gt_val + 0.02, f"GT {gt_val:.2f}", color='red')

        plt.xticks(x, metrics_for_bar, fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f"{model} - {language.upper()} MOS", fontsize=14)
        plt.ylim(0, max([s for s in scores if not (isinstance(s, float) and np.isnan(s))] + ([gt_val] if gt_val is not None else []) + [1.0]) * 1.15)
        if gt_val is not None:
            plt.legend()

        out_path = os.path.join(output_dir, f"bar_{language}_{model}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        saved.append(out_path)

    return saved


def plot_grouped_bars_by_model(metric_dict, mse_dict, language: str, output_dir: str):
    """单张图对比多个模型：每个指标四组柱，一个组里按模型并列；gtmos 画为多条水平虚线。
    在柱子上方标注 MSE 值。"""
    metrics_for_bar = ["ramp", "sslmos", "audiobox", "utmos"]
    models = [m for m in metric_dict.keys() if language in metric_dict[m]]
    if not models:
        print(f"语言 {language} 无可用数据")
        return None

    colors = ['#ffaf00', '#f46920', '#1f77b4', '#2ca02c']

    x = np.arange(len(metrics_for_bar))
    width = 0.8 / max(1, len(models))  # 留一点空隙

    plt.figure(figsize=(10, 6))

    # 只画 MSE 柱状图
    for i, model in enumerate(models):
        mse_data = mse_dict.get(model, {}).get(language, {})
        mse_scores = [mse_data.get(m, np.nan) for m in metrics_for_bar]
        offset = (i - (len(models)-1)/2) * width
        bars = plt.bar(x + offset, mse_scores, width=width, color=colors[i % len(colors)], alpha=0.9, label=model)
        # 柱顶部标注数值
        for b, s in zip(bars, mse_scores):
            if s is not None and not (isinstance(s, float) and np.isnan(s)):
                plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{s:.3f}",
                         ha='center', va='bottom', fontsize=10, color=colors[i % len(colors)], fontweight='bold')

    # y 上限
    all_vals = []
    for model in models:
        mse_data = mse_dict.get(model, {}).get(language, {})
        all_vals.extend([mse_data.get(m, np.nan) for m in metrics_for_bar])
    valid = [v for v in all_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    ymax = max(valid + [1.0]) * 1.15

    plt.xticks(x, metrics_for_bar, fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(f"MOS MSE Comparison ({language.upper()})", fontsize=14)
    plt.ylim(0, ymax)
    plt.legend()
    out_path = os.path.join(output_dir, f"bar_group_{language}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


if __name__ == "__main__":
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    # 只画 TTS 模型：可根据实际输出文件前缀过滤
    model_list = ["cosyvoice2", "xtts"]

    metric_dict, mse_dict = get_metric_dict(output_dir, model_list)
    print("Metric Dictionary:", metric_dict)
    print("\nMSE Dictionary:", mse_dict)

    # 保存指标和 MSE 给后续复用
    with open(os.path.join(output_dir, 'metric_tts_bar.json'), 'w', encoding='utf-8') as f:
        json.dump(metric_dict, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'mse_tts_bar.json'), 'w', encoding='utf-8') as f:
        json.dump(mse_dict, f, ensure_ascii=False, indent=2)

    languages = ['zh', 'en']

    # 只生成多模型同图（bar_group）
    for lang in languages:
        p = plot_grouped_bars_by_model(metric_dict, mse_dict, lang, output_dir)
        if p:
            print(f"saved: {p}")
