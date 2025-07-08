# TTS运行及评测工具


## 🚀 运行方法

### 第一部分：环境和模型准备

#### 1.1 环境配置

见 `Readm_downstream.md` 和 `Readm_upstream.md`

#### 1.2 模型下载

1. **RAMP模型检查点**：[ramp_ckpt](https://drive.google.com/file/d/1-l5huyOHWXFtSlGfHnHJVA7dcVS2RSdM/view?usp=sharing) 放置位置：`model_ckpt/ramp_ckpt`

2. **WavLM-large模型**：[wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing) 放置位置：项目根目录

3. **WavLM-large基础模型**：[wavlm_large.pt](https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt) 放置位置：项目根目录

4. **TTS模型** [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice), [XTTS](https://github.com/coqui-ai/TTS)

5. **ASR模型** Whisper-large-v3（英文ASR） Paraformer-zh（中文ASR）

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

# 国内镜像
# wget https://hf-mirror.com/coqui/XTTS-v2/resolve/main/model.pth
# wget https://hf-mirror.com/coqui/XTTS-v2/resolve/main/config.json
```

#### 2.2 数据说明

输入数据已包含在仓库的 `InputData/` 目录中：
- `InputData/en/meta.csv`：英文待合成任务信息
- `InputData/zh/meta.csv`：中文待合成任务信息
- `InputData/en/prompt-wavs`：英文参考音频
- `InputData/zh/prompt-wavs`：中文参考音频

格式说明：
```bash
# 待合成音频文本|参考音频路径|参考音频文本
infer_text|prompt_wav|prompt_text
```

#### 2.3 TTS运行

使用提供的脚本`run_upstream.sh`一键运行TTS语音合成:

**CosyVoice2模型：**
```bash
bash run_upstream.sh cosyvoice2
```

**XTTS模型：**
```bash
bash run_upstream.sh xtts
```

您也可以自定义模型，参考`Upstream/run_example.py`实现**example模型：**
```bash
bash run_upstream.sh example
```

**参数说明：**
- 模型名称（`cosyvoice2` 或 `xtts`或自定义`example`

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