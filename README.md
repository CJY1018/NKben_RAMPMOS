# Nkbench-eval: TTS/VC运行及自动化评测工具集成


## 🚀 TTS模块运行方法

### 第一部分：环境和模型准备

#### 1.1 模型下载

1. **RAMP模型检查点**：[ramp_ckpt](https://drive.google.com/file/d/1-l5huyOHWXFtSlGfHnHJVA7dcVS2RSdM/view?usp=sharing) 放置位置：`model_ckpt/ramp_ckpt`

2. **WavLM-large模型**：[wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing) 放置位置：项目根目录

3. **WavLM-large基础模型**：[wavlm_large.pt](https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt) 放置位置：项目根目录

4. **TTS模型** [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice), [XTTS](https://github.com/coqui-ai/TTS) 通过后续代码下载

5. **ASR模型** [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)（英文ASR） [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh)（中文ASR）项目自动下载

### 第二部分：上游运行（TTS语音合成）
#### 2.1 下载TTS模型
##### 2.1.1 下载[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)，放置在Upstream目录下
```bash
cd Upstream

# Clone the repo
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# Create Conda env
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Model download, Put it in Upstream/pretrained_models/CosyVoice2-0.5B
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
```

##### 2.1.2 下载[XTTS](https://github.com/coqui-ai/TTS)，放置在Upstream目录下
```bash
cd Upstream

# Clone the repo
git clone https://github.com/coqui-ai/TTS
cd TTS
pip install -e .[all]

# Create Conda env
conda create -n xtts -y python=3.10
conda activate xtts
pip install TTS

# Model download, Put it in Upstream/pretrained_models/XTTS-v2
cd Upstream/pretrained_models
mkdir XTTS-v2
cd XTTS-v2

wget https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json

# 国内镜像
# wget https://hf-mirror.com/coqui/XTTS-v2/resolve/main/model.pth
# wget https://hf-mirror.com/coqui/XTTS-v2/resolve/main/config.json
# wget https://hf-mirror.com/coqui/XTTS-v2/resolve/main/vocab.json
```

#### 2.2 数据准备和说明

输入数据已包含在仓库的 `InputData/` 目录中：
- `InputData/en/meta.csv`：英文待合成任务信息
- `InputData/zh/meta.csv`：中文待合成任务信息
- `InputData/en/prompt-wavs`：英文参考音频
- `InputData/zh/prompt-wavs`：中文参考音频

meta.csv格式说明：
```bash
# 待合成音频文本|参考音频路径|参考音频文本
infer_text|prompt_wav|prompt_text
```

#### 2.3 TTS运行

使用提供的脚本`run_upstream.sh`一键运行TTS语音合成:

**CosyVoice2模型：**
```bash
conda activate cosyvoice
bash run_upstream.sh cosyvoice2
```

**XTTS模型：**
```bash
conda activate xtts
bash run_upstream.sh xtts
```

您也可以自定义模型，参考`Upstream/run_example.py`实现**example模型：**
```bash
bash run_upstream.sh example
```
自定义example模型的过程见[第四部分](#第四部分自定义tts接入说明)

**参数说明：**
- 模型名称 (`cosyvoice2` 或 `xtts`或自定义`example`)

**输入结构：**
```bash
InputData/
├── en
│   ├── meta.csv            # 待合成任务的元信息
│   └── prompt-wavs         # 参考音频
└── zh
    ├── meta.csv            # 待合成任务的元信息
    └── prompt-wavs         # 参考音频
```

**输出结构：**
```bash
InputData/
├── en
│   ├── meta.csv
│   ├── meta_TTSMODEL.csv   # TTSMODEL合成音频的元信息
│   ├── prompt-wavs
│   └── wavs                # 合成的音频
└── zh
    ├── meta.csv
    ├── meta_TTSMODEL.csv   # TTSMODEL合成音频的元信息
    ├── prompt-wavs
    └── wavs                # 合成的音频
```

meta_TTSMODEL.csv 格式说明：
```bash
# 合成音频路径|合成音频文本|参考音频路径|参考音频文本
infer_wav|infer_text|prompt_wav|prompt_text

合成音频（infer_wav）的命名方式为：参考音频名称_TTSMODEL_uuid.wav
```

### 第三部分：下游运行（指标计算）
#### 3.1 配置环境
准备好使用Python 3.9.12和所需依赖项的环境：
```bash
conda create -n eval python=3.9.12
conda activate eval

# Clone and install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# Install additional requirements
pip install -r requirements.txt
```

#### 3.2 综合评估
使用提供的脚本`run_downstream.sh`一键运行所有评估：

**CosyVoice2模型：**
```bash
bash run_downstream.sh cosyvoice2
```

**XTTS模型：**
```bash
bash run_downstream.sh xtts
```

#### 3.3 输出结果
评估完成后，结果将输出在output文件夹中：
```bash
output
├── ramp_TTSMODEL_en.csv
├── ramp_TTSMODEL_zh.csv
├── similarity_TTSMODEL_en.csv
├── similarity_TTSMODEL_zh.csv
├── wer_TTSMODEL_en.csv
└── wer_TTSMODEL_zh.csv
```

#### 3.4 绘制指标雷达图
```bash
python run_plot.py
```
结果将输出在output文件夹中：
```bash
output
├── radar_chart_en.png
└── radar_chart_zh.png
```

输出示例图：
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/53f80e2d-f519-4631-8ebd-993dae58b5c4" alt="radar_chart_en"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/e02fd98a-5ce5-4e4c-b8c2-0ff5a7e0b3ab" alt="radar_chart_zh"/>
    </td>
  </tr>
  <tr>
    <td align="center">radar_chart_en.png</td>
    <td align="center">radar_chart_zh.png</td>
  </tr>
</table>


### 第四部分：自定义TTS接入说明
1. 模型准备：提供自定义TTS代码，置于到Upstream目录下，提供自定义的TTS模型到Upstream/pretrained_models/TTSMODEL目录下。若为API评测方式可跳过此步。
2. 参考`Upstream/run_example.py`中的TODO，实现自定义的`Upstream/run_MODEL_NAME.py`
    - 关键接口如下
        ```python
        # 根据提供的待合成文本（infer_text）、参考音频路径（prompt_wav_path）、参考音频文本（prompt_text）和语种（lang）实现自定义TTS的音频合成推理
        wav = tts.inference(infer_text, prompt_wav_path, prompt_text, lang)
        
        # 保存合成的音频到save_wav_path
        torchaudio.save(save_wav_path, wav, sample_rate)
        ```
3. 将模型名称MODEL_NAME添加到run_upstream.sh，脚本将执行`python Upstream/run_$MODEL_NAME.py --lang "$LANG"`
4. 运行`bash run_upstream.sh MODEL_NAME`启动上游TTS合成任务

## 🚀 VC模块运行方法
### 第一部分：上游运行（VC语音准换）
#### 1.1 下载[seed-vc](https://github.com/Plachtaa/seed-vc)项目，放置在Upstream_VC目录下
```bash
cd Upstream_VC

# Clone the repo
git clone https://github.com/Plachtaa/seed-vc.git
cd seed-vc

# Create Conda env
conda create -n seed-vc python=3.10
conda activate seed-vc
pip install -r requirements.txt
```

#### 1.2 数据准备和说明

输入数据已包含在仓库的 `InputData_VC/` 目录中：
- `InputData_VC/en/meta.csv`：英文待合成任务信息
- `InputData_CV/zh/meta.csv`：中文待合成任务信息
- `InputData_VC/en/source-wav`：英文原始音频
- `InputData_VC/zh/source-wav`：中文原始音频
- `InputData_VC/en/target-wav`: 英文目标音频
- `InputData_VC/zh/target-wav`: 中文目标音频

VC将把原始音频的音色转化为目标音频的音色

meta.csv格式说明：
```bash
# 原始音频路径|目标音频路径
source_wav|target_wav
```

#### 1.3 VC运行

使用提供的脚本`run_upstream_vc.sh`一键运行VC语音转换:

**seed-vc模型：**
```bash
conda activate seed-vc
bash run_upstream_vc.sh seed-vc
```
**注：** 运行seed-vc推理时，模型会自动下载到Upstream_VC/seed-vc/checkpoints目录

自定义example模型的过程见[第三部分](#第三部分自定义vc接入说明)

**参数说明：**
- 模型名称 (`seed-vc`或自定义`example`)
- 若需要修改VC默认参数，请见：Upstream_VC/run_seed-vc.py中模型参数说明

**输入结构：**
```bash
InputData_VC/
├── en
│   ├── meta.csv            # 待语音转换任务的元信息
│   └── source-wavs         # 原始音频
│   └── target-wavs         # 目标音频
└── zh
    ├── meta.csv            # 待语音转换任务的元信息
    ├── source-wavs         # 原始音频
    └── target-wavs         # 目标音频
```

**输出结构：**
```bash
InputData/
├── en
│   ├── meta.csv
│   ├── meta_VCMODEL.csv   # VCMODEL语音转换音频的元信息
│   ├── prompt-wavs
│   └── wavs                # 语音转换的音频
└── zh
    ├── meta.csv
    ├── meta_VCMODEL.csv   # VCMODEL语音转换音频的元信息
    ├── prompt-wavs
    └── wavs                # 语音转换的音频
```

meta_VCMODEL.csv 格式说明：
```bash
# 语音转换音频路径|原始音频路径|目标音频路径
infer_wav|source_wav|target_wav

# args参数说明见Upstream_VC/run_seed-vc.py
语音转换音频（infer_wav）的命名方式为：vc_v2_原始音频名称_目标音频名称_{args.length_adjust}_{args.diffusion_steps}_{args.similarity_cfg_rate}.wav
```

### 第二部分：下游运行（指标计算）
#### 2.1 配置环境
和[TTS下游运行评估环境](#31-配置环境)相同，仅需激活环境：
```bash
conda activate eval
```

#### 2.2 综合评估
相比于`TTS模型`输出结果wer、similarity和ramp三个维度的评估，`VC模型`输出结果仅对similarity和ramp两个维度进行评估

使用提供的脚本`run_downstream_vc.sh`一键运行所有评估：

**seed-vc模型：**
```bash
bash run_downstream_vc.sh seed-vc
```

#### 2.3 输出结果
评估完成后，结果将输出在output文件夹中：
```bash
output
├── ramp_VCMODEL_en.csv
├── ramp_VCMODEL_zh.csv
├── similarity_VCMODEL_en.csv
└── similarity_VCMODEL_zh.csv
```

### 第三部分：自定义VC接入说明
1. 模型准备：提供自定义VC代码，置于到Upstream_VC目录下。
2. 参考`Upstream_VC/run_example.py`中的TODO，实现自定义的`Upstream_VC/run_MODEL_NAME.py`
    - 关键接口如下
        ```python
        # 根据提供的原始音频路径（source_wav_path）、目标音频路径（target_wav_path）实现自定义VC的语音转换推理
        wav = vc_inference(source_wav_path, target_wav_path)
        
        # 保存合成的音频到save_wav_path
        torchaudio.save(save_wav_path, wav, sample_rate)
        ```
3. 将模型名称MODEL_NAME添加到run_upstream_vc.sh，脚本将执行`python Upstream_VC/run_$MODEL_NAME.py --lang "$LANG"`
4. 运行`bash run_upstream_vc.sh MODEL_NAME`启动上游VC语音转换任务