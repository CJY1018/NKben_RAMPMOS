import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_metric_dict(output_dir):
    metric_dict = {}  # 字典结构: {模型名: {语言: {指标: 值}}}

    for file in os.listdir(output_dir):
        if file.endswith('.csv'):
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
                    avg_wer = round(avg_wer * 100, 3)  # 转换为百分比并保留三位小数
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
                    
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")

    return metric_dict

# 绘制雷达图函数，包含归一化和半透明填充
def plot_radar_normalized(metric_dict, language):
    metrics = ['wer', 'ramp', 'similarity']
    models = list(metric_dict.keys())
    colors = ['#ffaf00', '#f46920']
    
    # 提取数据
    data = {model: [metric_dict[model][language][m] for m in metrics] for model in models}
    # 计算每个指标的最大值
    max_vals = np.max([vals for vals in data.values()], axis=0)
    
    # 角度分布
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection='polar')
    
    for i, (model, vals) in enumerate(data.items()):
        # 归一化
        norm_vals = np.array(vals) / max_vals
        norm_vals = norm_vals.tolist()
        norm_vals += norm_vals[:1]
        
        # 绘制曲线并填充面积
        ax.plot(angles, norm_vals, marker='o', label=model, color=colors[i])
        ax.fill(angles, norm_vals, alpha=0.25, color=colors[i])
    
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
        offset = 0.05  # 径向偏移量
        for j, (r, value) in enumerate(zip(label_radii, label_values)):
            # 计算标签的实际位置（径向稍微向外偏移）
            label_r = r + offset
            
            if j != len(label_radii) - 1:  # 最后一个标签不需要偏移
                # 添加文本标签
                ax.text(label_angles[j], label_r, value, 
                        ha='center', va='center', fontsize=8, color='dimgray')
            else:
                # 最后一个标签在轴上
                ax.text(label_angles[j], label_r + 0.05, value, 
                        ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    
    # 设置指标轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', pad=34)
    
    # 设置标题和图例
    plt.title(f'TTS Performance Comparison ({language.upper()})', fontsize=14, pad=20)
    plt.legend(loc='upper right')
    
    # 保存图像
    output_path = os.path.join(output_dir, f'radar_chart_{language}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()



if __name__ == "__main__":
    output_dir = 'output'  # 输出目录
    metric_dict = get_metric_dict(output_dir)
    
    print("Metric Dictionary:", metric_dict)
    
    languages = ['zh', 'en']
    
    # 分别绘制中文和英文的雷达图
    for lang in languages:
        plot_radar_normalized(metric_dict, lang)