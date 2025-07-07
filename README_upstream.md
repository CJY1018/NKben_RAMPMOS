# 上游模型准备

目前仓库中待集成cosyvoice+... (https://github.com/coqui-ai/TTS)

# 输入数据

InputData/en/meta.csv


**数据格式：**
```csv
待合成的音频ID名称|合成文本|提示音频路径|提示文本
common_voice_en_10119832_common_voice_en_103675-common_voice_en_103676|We asked over twenty different people, and they all said it was his.|prompt-wavs/common_voice_en_10119832.wav|One by one, the campfires were extinguished, and the oasis fell as quiet as the desert.
```

# model1: 集成说明

# model2: cosyvoice 运行说明

# 输出数据

OutputData/en/meta.csv

OutputData/en/wavs


**数据格式：**
```csv
合成的音频ID路径|合成文本|提示音频路径|提示文本
OutputData/en/common_voice_en_10119832_common_voice_en_103675-common_voice_en_103676|We asked over twenty different people, and they all said it was his.|prompt-wavs/common_voice_en_10119832.wav|One by one, the campfires were extinguished, and the oasis fell as quiet as the desert.
```