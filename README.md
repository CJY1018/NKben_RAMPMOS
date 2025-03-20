# 📈 **RAMP+: Retrieval-Augmented MOS Prediction with Prior Knowledge Integration**

Welcome to the official implementation of:

- **RAMP: Retrieval-Augmented MOS Prediction via Confidence-based Dynamic Weighting**
- **RAMP+: Retrieval-Augmented MOS Prediction with Prior Knowledge Integration**

This repository provides everything you need to evaluate and predict MOS (Mean Opinion Scores) efficiently with the **RAMP+ model**, leveraging **prior knowledge integration** to improve accuracy and handling out-of-domain (OOD) data gracefully. 

---

## 🚀 **Quick Evaluation Guide**

### 1. **Download Code and Checkpoint**

Get started by cloning the repository and downloading [the necessary checkpoint file](https://drive.google.com/file/d/1-l5huyOHWXFtSlGfHnHJVA7dcVS2RSdM/view?usp=sharing) to `NKben_RAMPMOS/model_ckpt`:

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

### 3. **Run Predictions**

Use `predict_ramp.py` to generate predictions. Just point to the checkpoint, datastore, and WAV file

```bash
python predict_ramp.py
```

For voice conversion evaluation, use `predict_ramp_VC.py`
```
python predict_ramp_VC.py
```

### Parameters:

- **`--checkpoint`**: The path to the downloaded model checkpoint. 

- **`--datastore_path`**: The path to the datastore. In this case, we have provided a BVCC datastore in `datasore_profile` as default, which makes it easier for you to evaluate the model. 

- **`--wavdir`**: The path to the directory containing the WAV files you want to predict on. 

- **`--outfile`**: The path where the prediction results will be saved. 


