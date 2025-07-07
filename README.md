# TTS运行及评测工具


## 🚀 运行方法

### 第一部分：环境和模型准备

#### 1.1 环境配置

见 `Readm_downstream.md` 和 `Readm_upstream.md`

#### 1.2 模型下载

1. **RAMP模型检查点**：[ramp_ckpt](https://drive.google.com/file/d/1-l5huyOHWXFtSlGfHnHJVA7dcVS2RSdM/view?usp=sharing) 放置位置：`model_ckpt/ramp_ckpt`

2. **WavLM-large模型**：[wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing) 放置位置：项目根目录

3. **WavLM-large基础模型**：[wavlm_large.pt](https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt) 放置位置：项目根目录

4. **ASR模型** Whisper-large-v3（英文ASR） Paraformer-zh（中文ASR）

### 第二部分：上游运行（TTS语音合成）

#### 2.1 数据说明

输入数据已包含在仓库的 `InputData/` 目录中：
- `InputData/en/meta.csv`：英文测试数据
- `InputData/zh/meta.csv`：中文测试数据


#### 2.2 TTS运行

使用提供的脚本运行TTS语音合成，目前cosyvoice, model2待集成：

**example模型：**
```bash
bash run_upstream.sh example output_example InputData
```

**CosyVoice模型：**
```bash
bash run_upstream.sh cosyvoice output_cosyvoice InputData
```



**参数说明：**
- 第一个参数：模型名称（`cosyvoice` 或 `example`）
- 第二个参数：输出路径
- 第三个参数：输入数据路径（可选，默认为 `InputData`）

**输出结构：**
```
output_cosyvoice/
├── zh
├───── meta.csv 
├───── wavs/         
├── en            # 合成的音频文件
├───── meta.csv 
└───── wavs/    
```

### 第三部分：下游运行（指标计算）

#### 3.2 综合评估

**一键运行所有评估：**
```bash
bash run_with_location.sh output_cosyvoice
```

#### 3.3 输出结果

评估完成后，结果文件说明: