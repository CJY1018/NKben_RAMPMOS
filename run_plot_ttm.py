import os
import json
import numpy as np
import torch
import subprocess
import time
import matplotlib.pyplot as plt
import csv
from msclap import CLAP

# ========== 配置部分 ==========
AUDIO_DIR = "InputData/ttm/wavs"
OUTPUT_JSON = "OutputData/ttm_eval/metrics_result.json"
OUTPUT_RADAR = "OutputData/ttm_eval/ttm_radar.png"
MOS_RESULT_PATH = "OutputData/ttm_eval/results.txt"
REFERENCE_SET = "fma_pop"  # fad 参考集
USE_GPU = True
PROMPT_PATH = "InputData/ttm/prompt_info.txt"


# ====== Step 1. 计算 FAD ======
def compute_fad_score(reference, audio_dir, timeout=600):
    cmd = ["fadtk", "vggish", reference, audio_dir]
    print("Running command:", " ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    fad_value = None
    start_time = time.time()
    last_line = ""
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(line)
            if "The FAD vggish score between" in last_line:
                try:
                    fad_value = float(line.split()[0])
                    print(f"✅ Detected FAD score: {fad_value}")
                except ValueError:
                    pass
            last_line = line
        if time.time() - start_time > timeout:
            process.kill()
            raise TimeoutError("FAD computation timed out.")
    process.stdout.close()
    process.wait()

    if fad_value is None:
        raise RuntimeError("❌ FAD score not found. Check fadtk output or model path.")
    return fad_value


# ====== Step 2. 计算 CLAP ======
def load_prompts():
    prompts = {}
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            prompts[row['id'].strip()] = row['text'].strip()
    return prompts


def compute_clap_score(audio_dir):
    clap_model = CLAP(version='2023', use_cuda=USE_GPU)
    prompts = load_prompts()
    scores = []

    for fname in os.listdir(audio_dir):
        if not fname.endswith(".wav"):
            continue
        fpath = os.path.join(audio_dir, fname)
        prompt_id = fname.split('_')[1].replace('.wav', '')
        prompt_text = prompts.get(f"{prompt_id}")
        if prompt_text is None:
            print(f"⚠️ Missing prompt for {fname}, skip")
            continue

        audio_emb = clap_model.get_audio_embeddings([fpath])
        text_emb = clap_model.get_text_embeddings([prompt_text])
        score = torch.nn.functional.cosine_similarity(audio_emb, text_emb).item()
        scores.append(score)

    if not scores:
        raise RuntimeError("❌ No valid CLAP scores computed.")
    avg_clap = float(np.mean(scores))*10    # 从[0,1]放缩到[0,10]，以和另三个指标对齐
    print(f"✅ CLAP average score: {avg_clap:.4f}")
    return avg_clap


# ====== Step 3. 读取 MOS 结果 ======
def read_mos_results(result_path):
    mos_quality, mos_consistency = None, None
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                mos_quality = float(parts[1])
                mos_consistency = float(parts[2])
            except ValueError:
                continue

    if mos_quality is None or mos_consistency is None:
        raise RuntimeError("❌ Failed to read MOS values from results.txt.")
    print(f"✅ MOS quality: {mos_quality}, MOS consistency: {mos_consistency}")
    return mos_quality, mos_consistency


# ====== Step 4. 绘制优化雷达图 ======
def plot_radar_optimized(metrics_dict, out_path):
    # 指标顺序和标签
    metrics = ['MOS (Quality)', 'MOS (Consistency)', 'FAD', 'CLAP']
    values = [
        metrics_dict['mos_quality'],
        metrics_dict['mos_consistency'],
        metrics_dict['fad'],
        metrics_dict['clap']
    ]

    # 角度分布
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 闭合数值
    vals = np.array(values + [values[0]])

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection='polar')
    ax.set_theta_offset(np.pi / 2)  # 0°在顶部
    ax.set_theta_direction(-1)      # 顺时针

    # 绘制雷达曲线和填充
    ax.plot(angles, vals, marker='o', color='#ffaf00', linewidth=2, label='musicgen-small')
    ax.fill(angles, vals, color='#ffaf00', alpha=0.25)

    # ===== 设置径向网格（自动计算最大值与步长） =====
    max_val = np.nanmax(vals)
    step = max_val / 5
    rgrids = np.arange(step, max_val + step, step)
    ax.set_rgrids(rgrids, angle=0, fontsize=8, color='lightgray')  # 保留默认一条轴刻度

    # ===== 手动在每个径向轴上绘制每个刻度值 =====
    for i, (metric) in enumerate(metrics):
        angle_rad = angles[i]
        label_radii = np.linspace(step, max_val, 5)  # 5个刻度位置
        label_values = [f"{r:.3f}" for r in label_radii]

        for j, (r, value) in enumerate(zip(label_radii, label_values)):
            # 在指标轴上添加径向刻度值
            ax.text(angle_rad, r, value,
                    ha='center', va='center',
                    fontsize=8, color='dimgray')

    # ===== 在每个指标轴顶端标出实际值 =====
    offset = step * 0.05
    for i, val in enumerate(values):
        ax.text(angles[i], val + offset, f"{val:.3f}",
                ha='center', va='center', fontsize=9, color='black', fontweight='bold')

    # ===== 设置指标标签 =====
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', pad=35)

    # ===== 标题与图例 =====
    plt.title("TTM Evaluation Radar Chart", fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=10)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Radar chart saved to: {out_path}")



# ====== Step 5. 主程序 ======
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_RADAR), exist_ok=True)

    fad_score = compute_fad_score(REFERENCE_SET, AUDIO_DIR)
    clap_score = compute_clap_score(AUDIO_DIR)
    mos_quality, mos_consistency = read_mos_results(MOS_RESULT_PATH)

    metrics = {
        "mos_quality": mos_quality,
        "mos_consistency": mos_consistency,
        "fad": fad_score,
        "clap": clap_score
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {OUTPUT_JSON}")

    plot_radar_optimized(metrics, OUTPUT_RADAR)
