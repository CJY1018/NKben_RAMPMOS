import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 主任务：比较 musiceval 与 audiobox（PQ scaled）在 TTM 任务上的 MSE
# ============================================================

def read_results_txt(path):
    """读取 musiceval 输出结果 (filename, score)"""
    data_overall = {}
    data_textual = {}
    if not os.path.exists(path):
        print(f"Musiceval results not found: {path}")
        return data_overall,data_textual

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            name = os.path.splitext(parts[0].strip())[0]
            try:
                score_overall = float(parts[1])  # 取 overall quality
                score_textual = float(parts[2])  # 取 textual alignment
            except ValueError:
                continue
            data_overall[name] = score_overall
            data_textual[name] = score_textual
    return data_overall,data_textual


def read_gt(path):
    """读取 Ground Truth MOS 列表"""
    gt_overall = {}
    gt_textual = {}
    if not os.path.exists(path):
        print(f"GT file not found: {path}")
        return gt_overall,gt_textual

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split(',')
            name = os.path.splitext(parts[0].strip())[0]
            if name.startswith('S010'):  # MusicEval 对应 S010 开头
                try:
                    score_overall = float(parts[1])
                    score_textual = float(parts[2])
                    gt_overall[name] = score_overall
                    gt_textual[name] = score_textual
                except ValueError:
                    continue
    return gt_overall,gt_textual


def compute_mse(pred_dict, gt_dict):
    """计算预测值与 GT 的 MSE"""
    common_keys = sorted(set(pred_dict) & set(gt_dict))
    if not common_keys:
        return None, 0
    preds = np.array([pred_dict[k] for k in common_keys], dtype=float)
    gts = np.array([gt_dict[k] for k in common_keys], dtype=float)
    mse = float(np.mean((preds - gts) ** 2))
    return round(mse, 6), len(common_keys)


def plot_mse_comparison(rows, out_path):
    """绘制两个方法的 MSE 对比柱状图（带上留白和 N/A 模型），风格与 grouped_bars 保持一致"""
    # 原有方法和 MSE
    labels = [r['method'] for r in rows]
    mses = [r['mse'] for r in rows]

    # 添加空模型
    extra_labels = ['audiobox', 'sslmos', 'utmos']
    extra_mses = [0, 0, 0]  # 占位
    all_labels = labels + extra_labels
    all_mses = mses + extra_mses

    # 配色：原有模型用指定颜色，空模型灰色
    base_colors = ['#ffaf00', '#f46920', '#1f77b4', '#2ca02c']
    colors = [base_colors[i % len(base_colors)] for i in range(len(labels))] + ['lightgray'] * len(extra_labels)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(all_labels))
    width = 0.6

    bars = plt.bar(x, all_mses, width=width, color=colors, alpha=0.85)

    # 顶部标注
    ymax = max(mses) * 1.15
    for i, (b, m) in enumerate(zip(bars, all_mses)):
        if i >= len(labels):  # 空模型
            plt.text(b.get_x() + b.get_width()/2, 0.02, "N/A",
                     ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        else:
            plt.text(b.get_x() + b.get_width()/2, m + ymax * 0.02, f"{m:.3f}",
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors[i])

    plt.xticks(x, all_labels, rotation=25, ha='right', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('MOS MSE Comparison (TTM)', fontsize=14)
    plt.ylim(0, ymax)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot to {out_path}")

if __name__ == "__main__":
    # 路径设置
    output_dir = 'OutputData/ttm_eval/'
    os.makedirs(output_dir, exist_ok=True)
    musiceval_results = 'OutputData/ttm_eval/results.txt'
    gt_file = 'InputData/ttm/total_mos_list.txt'
    wav_dir = 'InputData/ttm/wavs'

    # 1) 读取数据
    musiceval_overall, musiceval_textual = read_results_txt(musiceval_results)
    gt_overall, gt_textual = read_gt(gt_file)

    # 2) 计算Audiobox-PQ,与MusicEval-overall quality维度计算MSE作为对比
    audiobox_scores = {}
    if os.path.isdir(wav_dir):
        wav_paths = [os.path.join(wav_dir, fn) for fn in sorted(os.listdir(wav_dir)) if fn.endswith('.wav')]
        if wav_paths:
            try:
                from predict_audiobox import predict_audiobox
                meta_df = pd.DataFrame({'infer_wav': wav_paths})
                out_df, avg_audiobox = predict_audiobox(meta_df, metric='PQ')
                for _, row in out_df.iterrows():
                    name = os.path.splitext(str(row['filename']))[0]
                    audiobox_scores[name] = float(row['audiobox']) / 2.0  # 归一化到 1–5
                print(f"Audiobox predicted {len(audiobox_scores)} files, avg raw={avg_audiobox}")
            except Exception as e:
                print(f"⚠️ Failed to run audiobox prediction: {e}")
    else:
        print(f"⚠️ Wav dir not found: {wav_dir}")

    # 3) 计算 MSE
    musiceval_mse_overall, musiceval_n_overall = compute_mse(musiceval_overall, gt_overall)
    musiceval_mse_textual, musiceval_n_textual = compute_mse(musiceval_textual, gt_textual)
    audiobox_mse, audiobox_n = compute_mse(audiobox_scores, gt_overall)

    print(f"Musiceval MSE Overall={musiceval_mse_overall} (n={musiceval_n_overall})")
    print(f"Musiceval MSE Textual={musiceval_mse_textual} (n={musiceval_n_textual})")
    print(f"Audiobox MSE={audiobox_mse} (n={audiobox_n})")

    # 4) 保存overall quality结果 CSV
    rows_overall = []
    if musiceval_mse_overall is not None:
        rows_overall.append({'method': 'musiceval_overall', 'mse': musiceval_mse_overall, 'n': musiceval_n_overall})
    if audiobox_mse is not None:
        rows_overall.append({'method': 'audiobox_PQ_scaled', 'mse': audiobox_mse, 'n': audiobox_n})

    mse_df = pd.DataFrame(rows_overall)
    mse_csv = os.path.join(output_dir, 'ttm_mos_method_mse_overall.csv')
    mse_df.to_csv(mse_csv, index=False)
    print(f"✅ Saved MSE csv to {mse_csv}")

    # 5) 绘制overall quality MSE 柱状图
    out_png = os.path.join(output_dir, 'ttm_mos_methods_mse_overall.png')
    plot_mse_comparison(rows_overall, out_png)

    # 6) 保存textual 结果 CSV(仅musiceval单系统.无比对)
    rows_texual = []
    if musiceval_mse_textual is not None:
        rows_texual.append({'method': 'musiceval_textual', 'mse': musiceval_mse_textual, 'n': musiceval_n_textual})
        mse_df = pd.DataFrame(rows_overall)
    mse_csv = os.path.join(output_dir, 'ttm_mos_method_mse_textual.csv')
    mse_df.to_csv(mse_csv, index=False)
    print(f"✅ Saved MSE csv to {mse_csv}")
    
    # 7) 绘制 textual MSE 柱状图
    out_png = os.path.join(output_dir, 'ttm_mos_methods_mse_textual.png')
    plot_mse_comparison(rows_texual, out_png)

