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

#### 3.4 保存平均结果到本地 & 绘制指标雷达图
定义需要计算平均结果的模型
```bash
添加需要计算的模型名称到run_plot.py中的model_list
```

执行脚本
```bash
python run_plot.py
```
结果将输出在output文件夹中：
```bash
output
├── metric_tts.json
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

#### 2.4 保存平均结果到本地
定义需要计算平均结果的模型
```bash
添加需要计算的模型名称到run_result_vc.py中的model_list
```

执行脚本
```bash
python run_result_vc.py
```
结果将输出在output文件夹中：
```bash
output
└── metric_vc.json
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


---

# AI辅助评测工具2.0运行说明

## TTM自动质量评估模块
### 第一部分：环境和模型准备

#### 环境配置
```
python -V
# Python 3.10.16

# 安装依赖
pip install -r requirements_ttm.txt

# 安装audiocraft
pip install git+https://github.com/facebookresearch/audiocraft.git
```
#### 模型下载
- [CLAMP3预训练模型](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth)

- [下游预测头chechpoint](https://drive.google.com/drive/folders/1KJcUjfk4g1RCDp8sZ2n50vqHOu6Zce2V?usp=drive_link)：[MusicEval](https://arxiv.org/abs/2501.10811)中使用的基于3层MLP的预测头模型

#### 数据准备

用于音乐音频生成的文本输入位于 `InputData/ttm/prompt_info.txt`，每行包含一个文本提示词的 id 和文本内容


### 第二部分：上游运行（TTM文本生成音乐）

以`facebook/musicgen-small`为例，使用提供的脚本`run_upstream_ttm.sh`一键运行TTM文本生成音乐，生成的音频将位于 `InputData/ttm/wavs/`
```bash
bash run_upstream_ttm.sh
```

### 第三部分：下游运行（基于MusicEval的自动质量评估）

使用提供的脚本`run_downstream_ttm.sh`一键运行评估，包括：
- 文本拆分
- CLAMP3 文本/音频 global embed 提取
- 完成下游推理，保存结果到 `OutputData/ttm_eval/`

```bash
bash run_downstream_ttm.sh
```

### 第四部分：绘制TTM模型输出指标图

运行
```bash
python run_compute_mos_mse_ttm.py
python run_plot_ttm.py
```
结果将输出在OutputData/ttm_eval文件夹中：
```
OutputData/ttm_eval
├── metrics_result.json
├── ttm_mos_methods_mse_overall.csv
├── ttm_mos_methods_mse_overall.png
├── ttm_mos_methods_mse_textual.png
├── ttm_mos_methods_mse_textual.csv
└── ttm_radar.png
```
输出示例图：
<table>
  <tr>
    <td align="center">Overall Quality 维度</td>
    <td align="center">Textual Alignment 维度</td>
    <td align="center">各指标雷达图</td>
  </tr>
  <tr>
    <td align="center">
      <img src="OutputData/ttm_eval/ttm_mos_methods_mse.png" alt="ttm_mos_methods_mse"/>
    </td>
    <td align="center">
      <img src="OutputData/ttm_eval/ttm_mos_methods_mse_textual.png" alt="ttm_mos_methods_mse"/>
    </td>
    <td align="center">
      <img src="OutputData/ttm_eval/ttm_radar.png" alt="ttm_radar"/>
    </td>
  </tr>
  <tr>
    <td align="center">ttm_mos_methods_mse.png</td>
    <td align="center">ttm_mos_methods_mse_textual.png</td>
    <td align="center">ttm_radar.png</td>
  </tr>
</table>


## 🚀 更新：支持更多的MOS
除RAMP外，我们还集成了更多的MOS评测方法：[mos-finetune-ssl](https://github.com/nii-yamagishilab/mos-finetune-ssl)、[audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)和[UTMOS](https://github.com/sarulab-speech/UTMOS22)。

### 第一部分：环境准备
由于这3个MOS所需的环境与现有环境都有冲突，建议都新建一个环境进行安装
#### 1.1 [mos-finetune-ssl](https://github.com/nii-yamagishilab/mos-finetune-ssl)安装
```bash
# 在根目录创建MOS文件夹，以下3个MOS项目都放在该文件夹下
mkdir MOS && cd MOS
git clone https://github.com/nii-yamagishilab/mos-finetune-ssl.git

cd mos-finetune-ssl
conda env create -f environment.yml -n mos

# 注意：如果安装过程中报错，可能是两个原因：
# 1、 项目比较老，cuda版本使用11.1，如果您的机器cuda版本较新，可以先移除pytorch=1.9.0=py3.7_cuda11.1_cudnn8.0.5_0、cudatoolkit=11.1.1=h6406543_8、torchvision=0.10.0=py37_cu111、torchaudio==0.9.0，先安装其他的依赖，最后再单独安装pytorch和torchvision和torchaudio。或者参考根目录下的environment_mos.yml安装，使用cuda 11.7版本，实测在cuda 12下也可以运行
# 2、 pip的版本过高，我使用的是pip 21.1.2，使用pip install pip==21.1.2切换

# 配置fairseq，使用旧版本fairseq 0.10.2 (https://github.com/facebookresearch/fairseq/archive/refs/tags/v0.10.2.zip)

wget https://github.com/facebookresearch/fairseq/archive/refs/tags/v0.10.2.zip
unzip fairseq-0.10.2.zip # 确保解压到 MOS/mos-finetune-ssl/fairseq-0.10.2

# 下载模型：运行MOS/mos-finetune-ssl/run_inference.py
python run_inference.py
```

#### 1.2 [audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)安装
```bash
conda create -n audiobox python=3.10
conda activate audiobox
pip install audiobox_aesthetics

# 下载模型
# 首次运行会自动下载模型，下载的代码在MOS/audiobox-aesthetics/src/audiobox_aesthetics/utils.py，加载的代码在MOS/audiobox-aesthetics/src/audiobox_aesthetics/infer.py的134行，可自行修改为本地加载
```

#### 1.3 [UTMOS](https://github.com/sarulab-speech/UTMOS22)安装
```bash
git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo
cd UTMOS-demo

conda create -n UTMOS python=3.8
pip install -r requirements.txt
```

### 第二部分：MOS评测运行
#### 2.1 数据准备
和[TTS下游运行评估数据](#32-综合评估)相同，使用`InputData/en/`和`InputData/zh/`下的meta_TTSMODEL.csv和wavs文件夹

#### 2.2 综合评估
3个新加入的MOS评测均已无缝集成在`run_downstream.sh`脚本中，和之前的使用方式不变：

评估TTS模型的所有MOS指标：
```bash
bash run_downstream.sh cosyvoice2
bash run_downstream.sh xtts
```

评估VC模型的所有MOS指标：
```bash
bash run_downstream_vc.sh seed-vc
```

#### 2.3 输出结果
评估完成后，结果将输出在output文件夹中

TTS模型的输出结果：
```bash
output
├── sslmos_TTSMODEL_en.csv
├── ssltmos_TTSMODEL_zh.csv
├── audiobox_TTSMODEL_en.csv
├── audiobox_TTSMODEL_zh.csv
├── utmos_TTSMODEL_en.csv
└── utmos_TTSMODEL_zh.csv
```

VC模型的输出结果：
```bash
output
├── sslmos_VCMODEL_en.csv
├── sslmos_VCMODEL_zh.csv
├── audiobox_VCMODEL_en.csv
├── audiobox_VCMODEL_zh.csv
├── utmos_VCMODEL_en.csv
└── utmos_VCMODEL_zh.csv
```

#### 2.4 保存平均结果到本地 & 绘制指标雷达图
和[TTS/VC下游运行评估](#34-保存平均结果到本地--绘制指标雷达图)相同，直接运行`python run_plot.py`或`python run_result_vc.py`即可，结果将输出在output文件夹中：

TTS模型输出平均结果和雷达图：
```bash
python run_plot.py
```

VC模型输出平均结果：
```bash
python run_result_vc.py
```

## 🚀 更新：支持添加GroundTruth计算MSE
现在支持添加TTS模型和VC模型的GroundTruth MOS打分，用于计算各MOS模型和GT的MSE，显示在结果中，具体如下：

1. 准备GroundTruth MOS打分文件
在output文件夹下，按照gt_TTSMODEL_en.csv和gt_TTSMODEL_zh.csv或gt_VCMODEL_en.csv和gt_VCMODEL_zh.csv的格式准备好GroundTruth MOS打分文件，参考格式如下：
```csv
filename	gt_mos
10002287-00000094_cosyvoice2_c1d19433-a692-48c5-8a04-2f1cfa4e6bfd.wav	4.3
10002290-00000094_cosyvoice2_fb4cb112-a1c1-43a4-868f-a0c8ed124684.wav	4.1
10002352-00000030_cosyvoice2_e8f8f72d-0523-4432-ba70-a1c40d482442.wav	4.4
```

2. 重新结果输出
```bash
# TTS模型
python run_plot.py

# VC模型
python run_result_vc.py
```

计算MOS模型与GroundTruth MOS的MSE
```bash
python run_compute_mos_mse.py
```

结果和结果图将输出在output文件夹中：
+ output/mse_tts_bar.json
+ output/bar_group_en.png
+ output/bar_group_zh.png

输出示例图：
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/0187edb5-09c8-4921-8b41-060d850b929c" alt="radar_chart_en"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a178eda7-f5c0-4bbd-b674-1eb201fd90e4" alt="radar_chart_zh"/>
    </td>
  </tr>
  <tr>
    <td align="center">radar_chart_en.png</td>
    <td align="center">radar_chart_zh.png</td>
  </tr>
</table>
