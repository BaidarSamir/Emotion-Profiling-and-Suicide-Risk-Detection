# DistilBERT-Based Emotion Profiling and Suicide Risk Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.39+-yellow.svg)](https://huggingface.co/transformers/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

## Abstract

We investigate whether a compact Transformer (DistilBERT) can simultaneously support nuanced emotion profiling and binary suicide risk detection for text-based mental-health support systems. Using the GoEmotions corpus and a curated SuicideWatch dataset, we construct task-specific heads, apply class-weighted loss functions, and evaluate both validation and test performance. The study further quantifies label imbalance, monitors convergence behavior, and applies pruning plus dynamic quantization to produce deployable INT8 checkpoints. A production-ready FastAPI service exposes both models through REST endpoints for real-time inference.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Model Compression](#model-compression-and-deployment)
- [Installation & Usage](#installation--usage)
- [API Documentation](#api-documentation)
- [Reproduction](#reproduction-checklist)
- [Ethical Considerations](#ethical-considerations)
- [Citation](#citation)

## Repository Overview

```
.
├── Data/
│   ├── go_emotions_dataset.csv
│   └── Suicide_Detection.csv
├── emotions-and-suicide-risk-using-distilbert-model (2).ipynb
├── distilbert-emotion-suicide-risk.ipynb
├── app/
│   └── main.py
├── model_go/
│   ├── config.json
│   ├── model.safetensors
│   ├── model_go_pruned.pt
│   ├── model_go_pruned_quantized.pt
│   └── ...
├── model_sw/
│   ├── config.json
│   ├── model.safetensors
│   ├── model_sw_pruned.pt
│   ├── model_sw_pruned_quantized.pt
│   └── ...
└── visuals/
    └── training curves + evaluation plots (PNG)
```

## Datasets

- **GoEmotions** (Google/Pushshift): 211 225 Reddit comments annotated with 28 fine-grained emotions and `neutral`. We filter out `example_very_unclear` entries and discard samples without any positive emotion prior to running an iterative stratified split. Download: <https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset>.
- **SuicideWatch** (Nikhileswar Komati): 232 074 Reddit posts labeled as `suicide` or `non-suicide`. A stratified 10 000-post subset forms the training set; the remainder serves as held-out evaluation data. Download: <https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch>.

> Both corpora contain sensitive mental-health disclosures. Respect Kaggle licensing terms and handle all text responsibly.

### Dataset Snapshot

| Dataset | Rows × Cols | Key Columns | Notes |
| --- | --- | --- | --- |
| GoEmotions | 211 225 × 31 | `id`, `text`, `example_very_unclear`, 28 emotion indicators, `neutral` | 48.5 MB, boolean clarity flag + dense integer labels |
| SuicideWatch | 232 074 × 3 | `text`, `class` (`suicide` / `non-suicide`) | 5.3 MB, includes `Unnamed: 0` index column from Kaggle export |


### Emotion Label Distribution (10k GoEmotions Split)

| Label | Samples | Label | Samples |
| --- | --- | --- | --- |
| grief | 32 | disgust | 255 |
| relief | 62 | surprise | 265 |
| pride | 63 | excitement | 271 |
| nervousness | 87 | caring | 289 |
| embarrassment | 119 | sadness | 325 |
| remorse | 122 | confusion | 354 |
| fear | 154 | joy | 384 |
| desire | 184 | anger | 389 |
| love | 394 | gratitude | 559 |
| disappointment | 408 | annoyance | 655 |
| optimism | 419 | admiration | 824 |
| realization | 423 | approval | 848 |
| amusement | 445 | neutral | 2 661 |
| curiosity | 466 | — | — |

## Methodology

- **Pre-processing:** Remove unclear GoEmotions entries, enforce at least one positive label, and perform iterative multi-label stratification (`skmultilearn`). SuicideWatch texts are lowercased, stripped, and remapped to binary labels.
- **Modeling:** DistilBERT serves as a shared encoder. For GoEmotions we activate multi-label classification (`problem_type="multi_label_classification"`) and use a custom Trainer subclass with class-weighted `BCEWithLogitsLoss`. For SuicideWatch we fine-tune a binary head optimized with cross-entropy.
- **Evaluation Metrics:** Hamming score plus micro/macro F1 (GoEmotions) and accuracy/precision/recall/F1 (SuicideWatch). Sigmoid threshold defaults to 0.5 for every emotion logit.

## Architecture Overview

The study relies on DistilBERT's compressed Transformer backbone, which halves the number of layers relative to BERT-Base while retaining most of its representational capacity. The schematic below illustrates how the student network inherits the teacher's embeddings, multi-head attention, and feed-forward stacks, making it well-suited for efficient fine-tuning on safety-critical NLP tasks.

<div align="center">
  <img src="Diagram_of_BERT_BASE_and_Distil_BERT_model_architecture_facb5e7639.png" alt="DistilBERT vs. BERT architecture" width="800"/>
  <p><i>Figure 1: Comparison of BERT-Base and DistilBERT architectures. DistilBERT reduces model size by 40% while retaining 97% of BERT's performance.</i></p>
</div>

## Experimental Setup

- Hardware: single Tesla T4 GPU in Kaggle; the notebook auto-detects and falls back to CPU if unavailable.
- Tokenization: max length 128 (GoEmotions) and 256 (SuicideWatch) with static padding for efficient batching.
- Optimizer/TrainingArguments: batch size 8 for both tasks, GoEmotions trained for 10 epochs, SuicideWatch for 3 epochs, evaluation strategy = `epoch`, logging every 50 steps.
- Outputs: checkpoints, logs, and Matplotlib figures written to the workspace (e.g., `/kaggle/working` in Kaggle or the repo root locally).

## Results

### Quantitative Performance

| Task | Validation metrics | Test metrics |
| --- | --- | --- |
| **GoEmotions** | hamming = 0.3209, F1<sub>micro</sub> = 0.3640, F1<sub>macro</sub> = 0.3156, precision<sub>micro</sub> = 0.2975, recall<sub>micro</sub> = 0.4687 | hamming = 0.3209, F1<sub>micro</sub> = 0.3640, F1<sub>macro</sub> = 0.3156, precision<sub>micro</sub> = 0.2975, recall<sub>micro</sub> = 0.4687 |
| **SuicideWatch** | accuracy = 0.9740, F1 = 0.9739, precision = 0.9759, recall = 0.9720 | **accuracy = 0.9630**, **F1 = 0.9632**, precision = 0.9596, recall = 0.9667 |

### Training Curves

The training process demonstrates stable convergence for both tasks with minimal overfitting:

<div align="center">
  <img src="visuals/go_train_loss.png" width="400"/>
  <img src="visuals/go_val_loss.png" width="400"/>
  <p><i>Figure 2: GoEmotions training (left) and validation (right) loss curves over 10 epochs.</i></p>
</div>

<div align="center">
  <img src="visuals/go_eval_f1_micro.png" width="400"/>
  <img src="visuals/go_eval_f1_macro.png" width="400"/>
  <p><i>Figure 3: GoEmotions F1 scores (micro and macro) showing steady improvement across epochs.</i></p>
</div>

<div align="center">
## Installation & Usage

### Prerequisites

- Python 3.10 or higher
- 2GB disk space (models not included in repository)
- (Optional) CUDA-capable GPU for training

### Quick Start

```bash
# Clone the repository
git clone https://github.com/BaidarSamir/Emotion-Profiling-and-Suicide-Risk-Detection.git
cd Emotion-Profiling-and-Suicide-Risk-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Option 1: Train Models from Scratch

Run the Jupyter notebook to train both models (requires datasets):

```bash
# Install training dependencies
pip install jupyter matplotlib scikit-learn scikit-multilearn gdown

# Launch Jupyter
jupyter notebook distilbert-emotion-suicide-risk.ipynb
```

The notebook will:
1. Download GoEmotions and SuicideWatch datasets (or use local files from `Data/`)
2. Train both DistilBERT models (10k samples each)
3. Generate evaluation metrics and visualizations
4. Save models to `model_go/` and `model_sw/`
5. Create pruned and quantized versions for deployment

**Training time**: ~45 minutes on Tesla T4 GPU, ~4 hours on CPU

### Option 2: Use Pre-trained Models

Download pre-trained models from [Releases](#) (coming soon) and extract to project root:
- `model_go.zip` → extract to `model_go/`
- `model_sw.zip` → extract to `model_sw/`

### Start the API Server

```bash
# Ensure models exist in model_go/ and model_sw/
uvicorn app.main:app --reload
```

The server will be available at:
- **API**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

## API Documentation

### Endpoints

#### Health Check
```bash
GET /healthz
```
Returns server status and device information (CPU/GPU).

#### Emotion Profiling
```bash
POST /predict/emotions
Content-Type: application/json

{
  "text": "I am hopeful yet anxious about the future.",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "threshold": 0.5,
  "predictions": [
    {"label": "optimism", "probability": 0.7234},
    {"label": "nervousness", "probability": 0.6891}
  ],
  "full_distribution": [...]
}
```

#### Suicide Risk Detection
```bash
## Ethical Considerations

**Critical Safety Notice**: These models are research tools designed to augment, never replace, trained mental health professionals. 

### Deployment Guidelines

- ✅ **DO**: Use as an initial screening tool to identify at-risk content
- ✅ **DO**: Implement human-in-the-loop review for all high-risk classifications
- ✅ **DO**: Provide immediate access to crisis resources (e.g., 988 Suicide & Crisis Lifeline)
- ✅ **DO**: Maintain strict user privacy and obtain informed consent
- ✅ **DO**: Monitor for bias and regularly audit model performance across demographics

- ❌ **DON'T**: Make automated decisions without human oversight
- ❌ **DON'T**: Use as a diagnostic tool or replacement for clinical assessment
- ❌ **DON'T**: Deploy without proper crisis intervention protocols
- ❌ **DON'T**: Share user data without explicit consent

### Crisis Resources

- **US**: 988 Suicide & Crisis Lifeline | Text HOME to 741741
- **International**: https://www.iasp.info/resources/Crisis_Centres/

### Limitations

1. **Class Imbalance**: GoEmotions exhibits long-tail distribution (rare emotions have limited training samples)
2. **Cultural Context**: Models trained on English Reddit data may not generalize across languages or cultural contexts
3. **Temporal Drift**: Mental health language evolves; models require periodic retraining
4. **False Negatives**: High precision may sacrifice recall; some at-risk individuals may not be flagged

## Future Directions

1. **Expanded Training**: Increase GoEmotions coverage with class-balanced sampling to improve macro-level recall
2. **Multilingual Support**: Fine-tune on non-English corpora for broader accessibility
3. **Temporal Modeling**: Incorporate conversation history for context-aware risk assessment
4. **Explainability**: Integrate attention visualization and LIME/SHAP for interpretable predictions
5. **Calibration**: Apply temperature scaling and Platt scaling for reliable probability estimates

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{baidar2025emotion,
  author = {Baidar, Samir},
  title = {DistilBERT-Based Emotion Profiling and Suicide Risk Detection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/BaidarSamir/Emotion-Profiling-and-Suicide-Risk-Detection}
}
```

### Dataset Citations

**GoEmotions**:
```bibtex
@inproceedings{demszky2020goemotions,
  title={GoEmotions: A Dataset of Fine-Grained Emotions},
  author={Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```

**SuicideWatch**:
```bibtex
@misc{komati2021suicidewatch,
  author = {Komati, Nikhileswar},
  title = {Suicide Watch Dataset},
  year = {2021},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the Transformers library and DistilBERT implementation
- Google Research for the GoEmotions dataset
- Reddit communities for providing the training data
- FastAPI team for the excellent web framework

## Repository Structure

```
Emotion-Profiling-and-Suicide-Risk-Detection/
├── app/
│   └── main.py                          # FastAPI application with inference endpoints
├── visuals/                             # Training curves and evaluation plots
│   ├── go_train_loss.png
│   ├── go_val_loss.png
│   ├── go_eval_f1_micro.png
│   ├── go_eval_f1_macro.png
│   ├── go_eval_hamming_score.png
│   ├── sw_train_loss.png
│   ├── sw_val_loss.png
│   ├── sw_eval_accuracy.png
│   ├── sw_eval_f1.png
│   ├── sw_eval_precision.png
│   └── sw_eval_recall.png
├── distilbert-emotion-suicide-risk.ipynb  # Training notebook
├── Diagram_of_BERT_BASE_and_Distil_BERT_model_architecture_facb5e7639.png
├── Test.png                             # API testing screenshot
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── LICENSE                              # MIT License
├── .gitignore                           # Git ignore patterns
├── model_go/                            # GoEmotions model (not in repo - train or download)
└── model_sw/                            # SuicideWatch model (not in repo - train or download)
```

**Note**: Model directories (`model_go/`, `model_sw/`) and datasets (`Data/`) are excluded from the repository due to size constraints. Please train models using the notebook or download pre-trained versions.

## Contact

For questions, issues, or collaboration opportunities:
- **GitHub Issues**: [Create an issue](https://github.com/BaidarSamir/Emotion-Profiling-and-Suicide-Risk-Detection/issues)
- **Repository**: [BaidarSamir/Emotion-Profiling-and-Suicide-Risk-Detection](https://github.com/BaidarSamir/Emotion-Profiling-and-Suicide-Risk-Detection)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
  <p><b>⚠️ If you or someone you know is in crisis, please reach out for help immediately.</b></p>
  <p><b>US: Call/Text 988 | International: https://www.iasp.info/resources/Crisis_Centres/</b></p>
</div>
}
```

**Response:**
```json
{
  "label": "suicide",
  "confidence": 0.9234,
  "probabilities": {
    "non-suicide": 0.0766,
    "suicide": 0.9234
  }
}
```

### Python Client Example

```python
import requests

# Emotion analysis
response = requests.post(
    "http://localhost:8000/predict/emotions",
    json={"text": "I feel wonderful today!", "threshold": 0.5}
)
emotions = response.json()
print(f"Detected emotions: {[p['label'] for p in emotions['predictions']]}")

# Suicide risk assessment
response = requests.post(
    "http://localhost:8000/predict/suicide",
    json={"text": "I'm feeling stressed but managing."}
)
risk = response.json()
print(f"Risk level: {risk['label']} (confidence: {risk['confidence']:.2%})")
```

### Direct Model Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model_sw")
model = AutoModelForSequenceClassification.from_pretrained("./model_sw")
model.eval()

# For CPU deployment, use quantized version:
# model = torch.load("model_sw/model_sw_pruned_quantized.pt")

# Inference
text = "I can't stop thinking about hurting myself."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)
    
risk_score = probabilities[0, 1].item()  # Index 1 = "suicide"
print(f"Risk probability: {risk_score:.3f}")
```

For multi-label emotion classification, apply sigmoid activation:
```python
model_go = AutoModelForSequenceClassification.from_pretrained("./model_go")
logits = model_go(**inputs).logits
probabilities = torch.sigmoid(logits)  # Multi-label: sigmoid instead of softmax
```
# Suicide risk probability
curl -X POST http://localhost:8000/predict/suicide \
     -H "Content-Type: application/json" \
     -d '{"text":"I dont want to be here anymore."}'
```

Responses contain the probability distribution across all labels so downstream systems can build custom decision policies.

### Example Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = torch.load("model_sw/model_sw_pruned_quantized.pt")
model.eval()

inputs = tokenizer(
    "I can't stop thinking about hurting myself.",
    return_tensors="pt",
    truncation=True,
    padding=True
)
with torch.no_grad():
    logits = model(**inputs).logits
prob_suicide = torch.softmax(logits, dim=-1)[0, 1].item()
print(f"Risk probability: {prob_suicide:.3f}")
```

Use the GoEmotions checkpoint with a sigmoid activation to obtain 29 probability scores and adjust decision thresholds per label if required.

## Reproduction Checklist

1. Create a virtual environment (Python ≥ 3.10) and install `torch`, `transformers`, `scikit-learn`, `scikit-multilearn`, `pandas`, `numpy`, `jupyter`.
2. Place the Kaggle CSVs in `Data/` or update the notebook paths.
3. Launch Jupyter (`jupyter notebook`) and run `emotions-and-suicide-risk-using-distilbert-model (2).ipynb` sequentially (the original notebook remains for reference).
4. Update `output_dir` and checkpoint paths if running outside Kaggle to avoid writing to `/kaggle/working`.

## Ethical Considerations

These models analyze text about self-harm and should only augment, never replace, trained professionals. Deployments must incorporate human-in-the-loop review, clear escalation policies, and guardrails against misuse. Maintain user privacy and obtain all necessary approvals before processing sensitive content.

## Future Directions

1. Increase GoEmotions coverage (larger training subset or class-balanced sampling) to improve macro-level recall.
2. Integrate experiment tracking (Weights & Biases, MLflow) for hyper-parameter sweeps.
3. Wrap the quantized checkpoints in a FastAPI/Gradio service with confidence calibration and explanation tooling.
