# 📈 **RAMP+: Retrieval-Augmented MOS Prediction with Prior Knowledge Integration**

Welcome to the official implementation of:

- **RAMP: Retrieval-Augmented MOS Prediction via Confidence-based Dynamic Weighting**
- **RAMP+: Retrieval-Augmented MOS Prediction with Prior Knowledge Integration**

This repository provides everything you need to evaluate and predict MOS (Mean Opinion Scores) efficiently with the **RAMP+ model**, leveraging **prior knowledge integration** to improve accuracy and handling out-of-domain (OOD) data gracefully. 

---

## 🚀 **Quick Evaluation Guide**

### 1. **Download Code and Checkpoint**

Get started by cloning the repository and downloading [the necessary checkpoint file](https://drive.google.com/file/d/1-l5huyOHWXFtSlGfHnHJVA7dcVS2RSdM/view?usp=sharing) to `NKben_RAMPMOS/model_ckpt`:

For WER, we employ [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) as the automatic speech recognition (ASR) engines for English and Mandarin, respectively. The model will be downloaded automatically.

For SIM, we use WavLM-large fine-tuned on the speaker verification task [model link](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view) to obtain speaker embeddings used to calculate the cosine similarity of speech samples of each test utterance against reference clips. And you should download [wavlm_large.pt](https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt), both of these two models are placed in the directory `NKben_RAMPMOS` directory.

For Voice Concersion(VC) task, download [datastore_profile_vc](https://drive.google.com/drive/folders/12dmLPFIB8V2-lV3yTxfI4h4Eo6pLR5WP?usp=drive_link) to `NKben_RAMPMOS` and download [`ramp_ckpt_vc`]() to `NKben_RAMPMOS/model_ckpt`. 

### 2. **Set Up the Environment**

Get the environment ready to go with Python 3.9.12 and required dependencies:

```bash
conda create -n RAMP python=3.9.12
conda activate RAMP

# Clone and install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# Install additional requirements
pip install -r requirements.txt
```

### 3. **Prepare meta.csv and audios**
The test set is mainly organized using the method of meta file. The meaning of each line in the `meta.csv`: 
```bash
the filename of the synthesized audio | the text of the synthesized audio | the filename of the prompt audio | the text of the prompt audio.
```

Organize the files in the following way: place the synthesized audio in the wavs directory and the reference audio in the prompt_wavs directory:
```bash
.
├── meta.csv
├── prompt_wavs
│   ├── 10002287-00000094.wav
│   └── 10002290-00000094.wav
└── wavs
    ├── 10002287-00000095.wav
    ├── 10002287-00000099.wav
    └── 10002290-00000102.wav
```
You can refer to `NKben_RAMPMOS/new_example_samples` directory as an example.

### 4. **Run Predictions**

Use `run_wer.py` to generate wer predictions. Point to the path of meta.csv, and language of the text (en or zh).
```bash
python run_wer.py --meta_csv path/to/meta.csv --lang en/zh
```

Use `run_sim.py` to generate similarity predictions. Point to the path of meta.csv, and the path to the sim model checkpoint(wavlm_large_finetune.pth).
```bash
python run_sim.py --meta_csv path/to/meta.csv --sim_checkpoint_path path/to/wavlm_large_finetune.pth
```

Use `run_ramp.py` to generate wer predictions. Point to the path of meta.csv, the path to the RAMP model checkpoint(ramp_ckpt) and the path to the datastore for RAMP.
```bash
python run_ramp.py --meta_csv path/to/meta.csv --ramp_checkpoint_path path/to/ramp_ckpt --datastore_path path/to/datastore_profile
```

Use `run_all.py` to generate wer, sim and ramp predictions. Point to all the parameters mentioned above.
```bash
python run_all.py --meta_csv path/to/meta.csv --lang en/zh --sim_checkpoint_path path/to/wavlm_large_finetune.pth --ramp_checkpoint_path path/to/ramp_ckpt --datastore_path path/to/datastore_profile
```

The predicted output csv files will be generated at `NKben_RAMPMOS/ouput`.

For voice conversion evaluation, use `predict_ramp_VC.py`
```
python predict_ramp_VC.py
```

### Parameters:
- **`--meta_csv`**: The path to the meta CSV file.

- Only for run_wer.py:
    - **`--lang`**: Language of the text (en or zh).

- Only for run_sim.py:
    - **`--sim_checkpoint_path`**: The path to the sim model checkpoint.

- Only for run_sim.py:
    - **`--ramp_checkpoint_path`**: The path to the RAMP model checkpoint.
    - **`--datastore_path`**: The path to the datastore for RAMP.

- For predict_ramp.py and predict_ramp_VC.py(Old):
    - **`--checkpoint`**: The path to the downloaded model checkpoint. 

    - **`--datastore_path`**: The path to the datastore. In this case, we have provided a BVCC datastore in `datasore_profile` as default, which makes it easier for you to evaluate the model. 

    - **`--wavdir`**: The path to the directory containing the WAV files you want to predict on. 

    - **`--outfile`**: The path where the prediction results will be saved. 


