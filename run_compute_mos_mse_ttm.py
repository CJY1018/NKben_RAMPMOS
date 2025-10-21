import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 主任务：比较 musiceval 与 audiobox（PQ scaled）在 TTM 任务上的 MSE
# ============================================================

def read_results_txt(path):
    """读取 musiceval 输出结果 (filename, score)"""
    data = {}
    if not os.path.exists(path):
        print(f"Musiceval results not found: {path}")
        return data

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            name = os.path.splitext(parts[0].strip())[0]
            try:
                score = float(parts[1])  # 取 overall quality
            except ValueError:
                continue
            data[name] = score
    return data


def read_gt(path):
    """读取 Ground Truth MOS 列表"""
    data = {}
    if not os.path.exists(path):
        print(f"GT file not found: {path}")
        return data

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split(',')
            name = os.path.splitext(parts[0].strip())[0]
            if name.startswith('S010'):  # MusicEval 对应 S010 开头
                try:
                    score = float(parts[1])
                    data[name] = score
                except ValueError:
                    continue
    return data


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
    """绘制两个方法的 MSE 对比柱状图（带上留白）"""
    labels = [r['method'] for r in rows]
    mses = [r['mse'] for r in rows]

    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))
    bars = plt.bar(x, mses, color=['#1f77b4', '#ff7f0e'][:len(labels)], alpha=0.9)

    # 顶部标注并自动留白
    ymax = max(mses)
    for b, m in zip(bars, mses):
        plt.text(b.get_x() + b.get_width() / 2, m + ymax * 0.05, f"{m:.6f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylim(0, ymax * 1.25)
    plt.xticks(x, labels, rotation=25, ha='right', fontsize=11)
    plt.ylabel('MSE vs GT', fontsize=12)
    plt.title('TTM MSE Comparison: musiceval vs audiobox (PQ scaled)', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot to {out_path}")


if __name__ == "__main__":
    # 路径设置
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    musiceval_results = 'OutputData/ttm_eval/results.txt'
    gt_file = 'InputData/ttm/total_mos_list.txt'
    wav_dir = 'InputData/ttm/wavs'

    # 1) 读取数据
    musiceval = read_results_txt(musiceval_results)
    gt = read_gt(gt_file)

    # 2) 运行 Audiobox 预测（如可用）
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
    musiceval_mse, musiceval_n = compute_mse(musiceval, gt)
    audiobox_mse, audiobox_n = compute_mse(audiobox_scores, gt)

    print(f"Musiceval MSE={musiceval_mse} (n={musiceval_n})")
    print(f"Audiobox MSE={audiobox_mse} (n={audiobox_n})")

    # 4) 保存结果 CSV
    rows = []
    if musiceval_mse is not None:
        rows.append({'method': 'musiceval', 'mse': musiceval_mse, 'n': musiceval_n})
    if audiobox_mse is not None:
        rows.append({'method': 'audiobox_PQ_scaled', 'mse': audiobox_mse, 'n': audiobox_n})

    mse_df = pd.DataFrame(rows)
    mse_csv = os.path.join(output_dir, 'ttm_mos_method_mse.csv')
    mse_df.to_csv(mse_csv, index=False)
    print(f"✅ Saved MSE csv to {mse_csv}")

    # 5) 绘制图像
    out_png = os.path.join(output_dir, 'ttm_mos_methods_mse.png')
    plot_mse_comparison(rows, out_png)

    # 6) 导出对齐数据便于检查
    all_keys = sorted(set(gt) | set(musiceval) | set(audiobox_scores))
    aligned_rows = [{
        'filename': k,
        'gt': gt.get(k, np.nan),
        'musiceval': musiceval.get(k, np.nan),
        'audiobox_PQ_scaled': audiobox_scores.get(k, np.nan)
    } for k in all_keys]
    df_aligned = pd.DataFrame(aligned_rows)
    aligned_csv = os.path.join(output_dir, 'ttm_aligned_predictions.csv')
    df_aligned.to_csv(aligned_csv, index=False)
    print(f"✅ Saved aligned predictions to {aligned_csv}")
