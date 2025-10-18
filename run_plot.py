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
                
                # 根据指标类型存储结果
                if metric == 'wer':
                    word_counts = df['infer_text'].str.split().apply(len)
                    avg_wer = (df['wer'] * word_counts).sum() / word_counts.sum() # 微平均 (Micro‑average)／加权平均，不影响中文计算
                    avg_wer = round(avg_wer, 3)  # 保留三位小数
                    # avg_wer = df['wer'].mean()  # 宏平均 (Macro-average)
                    metric_dict[model_name][lang]['wer'] = avg_wer
                elif metric == 'similarity':
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
                    avg_ramp = df['gt_mos'].mean() # 宏平均 (Macro-average)
                    avg_ramp = round(avg_ramp, 3)
                    metric_dict[model_name][lang]['gt_mos'] = avg_ramp
                    
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")

    return metric_dict

# 绘制雷达图函数，包含归一化和半透明填充
def plot_radar_normalized(metric_dict, language):
    metrics = ['wer', 'similarity', 'ramp', 'sslmos', 'audiobox', 'utmos']
    models = list(metric_dict.keys())
    colors = ['#ffaf00', '#f46920']
    
    # 提取数据
    data = {model: [metric_dict[model][language].get(m, np.nan) for m in metrics] for model in models}

    # 如果存在GT（每个模型的gt分数），则提取GT用于在特定指标上绘制红色虚线
    # GT 只用于 ramp/sslmos/audiobox/utmos 四个指标的位置
    gt_indices = [metrics.index(x) for x in ['ramp', 'sslmos', 'audiobox', 'utmos']]
    gt_vals_by_model = {}
    for model in models:
        gt_val = metric_dict[model][language].get('gt_mos', None) if language in metric_dict[model] else None
        # 如果没有gt，设置为None
        gt_vals_by_model[model] = gt_val

    # 计算每个指标的最大值（归一化基准），同时把GT值考虑在内以保证GT线可见
    stacked = []
    for model in models:
        vals = np.array([v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0.0 for v in data[model]])
        stacked.append(vals)
        # 将gt放到对应指标位置（如果存在）以纳入最大值计算
        gt_val = gt_vals_by_model.get(model)
        if gt_val is not None:
            gt_arr = np.zeros(len(metrics))
            for idx in gt_indices:
                # audiobox 的 GT 是双倍尺度（数据表上定义），因此在 audiobox 索引上使用 gt_val*2
                if metrics[idx] == 'audiobox':
                    gt_arr[idx] = gt_val * 2
                else:
                    gt_arr[idx] = gt_val
            stacked.append(gt_arr)

    max_vals = np.max(np.vstack(stacked), axis=0)
    # 避免除以0
    max_vals[max_vals == 0] = 1.0
    
    # 角度分布
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection='polar')
    
    for i, (model, vals) in enumerate(data.items()):
        # 归一化
        vals_arr = np.array([v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0.0 for v in vals])
        norm_vals = vals_arr / max_vals
        norm_vals = norm_vals.tolist()
        norm_vals += norm_vals[:1]
        
        # 绘制曲线并填充面积
        ax.plot(angles, norm_vals, marker='o', label=model, color=colors[i])
        ax.fill(angles, norm_vals, alpha=0.25, color=colors[i])

    # 绘制GT红色虚线（仅在ramp/sslmos/audiobox/utmos处有值）
    # 我们用每个模型的GT值绘制一条相同样式的线（红色虚线），若不同模型有不同GT也都绘制
    for i, model in enumerate(models):
        gt_val = gt_vals_by_model.get(model)
        if gt_val is None:
            continue
        # 构造GT在所有指标上的值（只有指定指标有gt，其他为0）
        gt_full = np.zeros(len(metrics))
        for idx in gt_indices:
            # audiobox 的 GT 使用两倍
            if metrics[idx] == 'audiobox':
                gt_full[idx] = gt_val * 2
            else:
                gt_full[idx] = gt_val
        # 归一化
        gt_norm = gt_full / max_vals

        # 为避免从一个非零点连接到另一个非零点穿过0（尤其是当某些指标为0/NaN时），
        # 我们按连续非零区间分段绘制GT线，支持环绕（wrap-around）
        nonzero = np.where(gt_norm != 0)[0].tolist()
        if not nonzero:
            continue

        # helper: 获取连续区间（考虑环绕）
        segments = []
        start = nonzero[0]
        prev = nonzero[0]
        for idx in nonzero[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            else:
                segments.append((start, prev))
                start = idx
                prev = idx
        segments.append((start, prev))

        # 处理环绕：若首尾相连则合并
        if len(segments) > 1 and segments[0][0] == 0 and segments[-1][1] == len(metrics) - 1:
            segments[0] = (segments[-1][0], segments[0][1])
            segments.pop()

        # 绘制每个段。对于每个段，我们需要构造角度和对应的值，注意不能闭合（不添加重复点）
        for (s, e) in segments:
            if s <= e:
                seg_idxs = list(range(s, e + 1))
            else:
                # 环绕段例如从 4 到 1（不常见，因为我们已合并首尾），构造跨边索引
                seg_idxs = list(range(s, len(metrics))) + list(range(0, e + 1))

            seg_angles = [angles[idx] for idx in seg_idxs]
            seg_vals = [float(gt_norm[idx]) for idx in seg_idxs]

            # 绘制分段虚线
            ax.plot(seg_angles, seg_vals, linestyle='--', color=colors[i], label=f'GT ({model})' if s == segments[0][0] else None, linewidth=2)
    
    # 设置径向网格线
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_yticklabels([]) # 去掉径向标签
    
    # 在每个指标轴上添加实际值标签
    for i, (metric, max_val) in enumerate(zip(metrics, max_vals)):
        angle_rad = angles[i]
        
        # 计算标签位置（在指标轴线上）
        label_angles = np.linspace(angle_rad, angle_rad, 5)  # 5个刻度位置
        label_radii = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        label_values = [f"{max_val * r:.3f}" for r in label_radii]
        
        # 添加标签（稍微偏移以避免重叠）
        offset = 0.03  # 径向偏移量
        for j, (r, value) in enumerate(zip(label_radii, label_values)):
            # 计算标签的实际位置（径向稍微向外偏移）
            label_r = r + offset
            
            if j != len(label_radii) - 1:  # 最后一个标签不需要偏移
                # 添加文本标签
                ax.text(label_angles[j], label_r, value, 
                        ha='center', va='center', fontsize=8, color='dimgray')
            else:
                # 最后一个标签在轴上
                ax.text(label_angles[j], label_r + 0.03, value, 
                        ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    
    # 设置指标轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', pad=40)
    
    # 设置标题和图例
    plt.title(f'TTS Performance Comparison ({language.upper()})', fontsize=14, pad=20)
    plt.legend(loc='upper right')
    
    # 保存图像
    output_path = os.path.join(output_dir, f'radar_chart_{language}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()



if __name__ == "__main__":
    output_dir = 'output'  # 输出目录
    model_list = ["cosyvoice2", "xtts"] # 定义需要计算的模型
    metric_dict = get_metric_dict(output_dir, model_list)
    
    print("Metric Dictionary:", metric_dict)
    
    # 保存metric_dict为JSON文件
    metric_list = [{model: metric_dict[model]} for model in metric_dict ]
    with open(os.path.join(output_dir, 'metric_tts.json'), 'w', encoding='utf-8') as f:
        json.dump(metric_list, f, ensure_ascii=False, indent=4)

    languages = ['zh', 'en']
    
    # 分别绘制中文和英文的雷达图
    for lang in languages:
        plot_radar_normalized(metric_dict, lang)