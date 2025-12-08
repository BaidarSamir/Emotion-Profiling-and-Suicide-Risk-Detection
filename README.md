# DistilBERT-Based Emotion Profiling and Suicide Risk Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.39+-yellow.svg)](https://huggingface.co/transformers/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

## Abstract

This study implements DistilBERT-based models for two distinct mental health classification tasks: multi-label emotion profiling on the GoEmotions dataset (29 categories) and binary suicide risk detection on the SuicideWatch dataset. The implementation incorporates Focal Loss to address class imbalance, class-specific threshold optimization for multi-label predictions, and stratified 40/10/50 train/validation/test splits. On held-out test sets, models achieve 37.66% F1-micro and 32.50% F1-macro on GoEmotions (103,877 samples), and 97.08% accuracy with 97.26% recall on SuicideWatch (116,037 samples), demonstrating strong generalization with minimal validation-test performance gaps. The study includes model compression via 30% pruning and INT8 quantization for efficient deployment, with a FastAPI service providing REST endpoints for real-time inference.

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

## Repository Structure

```
Emotion-Profiling-and-Suicide-Risk-Detection/
│
├──  Notebooks
│   ├── distilbert-emotion-suicide-risk-improved.ipynb  # Main training notebook (Focal Loss + improvements)
│
├──  Data (not in repo - download from Kaggle)
│   ├── go_emotions_dataset.csv                         # GoEmotions: 211K Reddit comments
│   └── Suicide_Detection.csv                           # SuicideWatch: 232K Reddit posts
│
├──  Models (not in repo - train via notebook or download)
│   ├── model_go/                                       # GoEmotions emotion classifier
│   │   ├── config.json                                 # Model configuration
│   │   ├── model.safetensors                           # FP32 weights (265 MB)
│   │   ├── optimal_thresholds.npy                      # Per-class decision thresholds
│   │   ├── model_go_pruned.pt                          # 30% pruned (186 MB)
│   │   ├── model_go_pruned_quantized.pt                # Pruned + INT8 (68 MB)
│   │   ├── tokenizer_config.json                       # Tokenizer settings
│   │   ├── tokenizer.json                              # Fast tokenizer vocabulary
│   │   ├── vocab.txt                                   # WordPiece vocabulary
│   │   └── special_tokens_map.json                     # Special tokens mapping
│   │
│   └── model_sw/                                       # SuicideWatch risk detector
│       ├── config.json                                 # Model configuration
│       ├── model.safetensors                           # FP32 weights (265 MB)
│       ├── model_sw_pruned.pt                          # 30% pruned (186 MB)
│       ├── model_sw_pruned_quantized.pt                # Pruned + INT8 (68 MB)
│       ├── tokenizer_config.json                       # Tokenizer settings
│       ├── tokenizer.json                              # Fast tokenizer vocabulary
│       ├── vocab.txt                                   # WordPiece vocabulary
│       └── special_tokens_map.json                     # Special tokens mapping
│
├──  API Server
│   ├── app/
│   │   └── main.py                                     # FastAPI REST endpoints
│   └── requirements.txt                                # Python dependencies
│
├──  Visualization Assets
│   └── visuals/                                        # Training curves and plots
│       ├── go_train_loss.png                           # GoEmotions training loss
│       ├── go_val_loss.png                             # GoEmotions validation loss
│       ├── go_eval_f1_micro.png                        # GoEmotions F1-micro progression
│       ├── go_eval_f1_macro.png                        # GoEmotions F1-macro progression
│       ├── go_eval_hamming_score.png                   # GoEmotions Hamming score
│       ├── sw_train_loss.png                           # SuicideWatch training loss
│       ├── sw_val_loss.png                             # SuicideWatch validation loss
│       ├── sw_eval_accuracy.png                        # SuicideWatch accuracy progression
│       ├── sw_eval_f1.png                              # SuicideWatch F1-score progression
│       ├── sw_eval_precision.png                       # SuicideWatch precision progression
│       └── sw_eval_recall.png                          # SuicideWatch recall progression
│
├──  Documentation
│   ├── README.md                                       # This file
│   ├── LICENSE                                         # MIT License
│   ├── Diagram_of_BERT_BASE_and_Distil_BERT_model_architecture_facb5e7639.png
│   └── Test.png                                        # API testing screenshot
│
└──  Configuration
    └── .gitignore                                      # Git ignore patterns
```

### File Size Summary

| Component | Size | Notes |
| --- | --- | --- |
| **Datasets** | 53.8 MB | GoEmotions: 48.5 MB, SuicideWatch: 5.3 MB |
| **Full Models (FP32)** | 530 MB | Both models (265 MB each) |
| **Pruned Models (30%)** | 372 MB | Both models (186 MB each) |
| **Quantized Models (INT8)** | 136 MB | Both models (68 MB each) |
| **Training Curves** | ~2 MB | 11 PNG visualization files |
| **Total (with all variants)** | ~1.1 GB | Excluding datasets |

> **Note:** Model directories (`model_go/`, `model_sw/`) and datasets (`Data/`) are excluded from Git repository due to size constraints. Train models using the notebook or download pre-trained versions from releases.

## Datasets

- **GoEmotions** (Google/Pushshift): 211,225 Reddit comments annotated with 28 fine-grained emotions and `neutral`. After filtering `example_very_unclear` entries and removing samples without positive emotion labels, the dataset is split using iterative stratified sampling into 83,150 training / 20,787 validation / 103,877 test samples (40/10/50 split). Download: <https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset>.
- **SuicideWatch** (Nikhileswar Komati): 232,074 Reddit posts labeled as `suicide` or `non-suicide`. The dataset is stratified into 92,829 training / 23,208 validation / 116,037 test samples (40/10/50 split). Class distribution is balanced at approximately 50/50. Download: <https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch>.

> Both corpora contain sensitive mental-health disclosures. Respect Kaggle licensing terms and handle all text responsibly.

### Dataset Snapshot

| Dataset | Rows × Cols | Key Columns | Notes |
| --- | --- | --- | --- |
| GoEmotions | 211 225 × 31 | `id`, `text`, `example_very_unclear`, 28 emotion indicators, `neutral` | 48.5 MB, boolean clarity flag + dense integer labels |
| SuicideWatch | 232 074 × 3 | `text`, `class` (`suicide` / `non-suicide`) | 5.3 MB, includes `Unnamed: 0` index column from Kaggle export |


### Emotion Label Distribution (83k GoEmotions Training Split)

| Label | Samples | Label | Samples |
| --- | --- | --- | --- |
| grief | 270 | disgust | 2,120 |
| relief | 515 | surprise | 2,206 |
| pride | 521 | excitement | 2,252 |
| nervousness | 724 | caring | 2,399 |
| embarrassment | 990 | sadness | 2,703 |
| remorse | 1,010 | confusion | 2,944 |
| fear | 1,279 | joy | 3,194 |
| desire | 1,527 | anger | 3,234 |
| love | 3,277 | gratitude | 4,645 |
| disappointment | 3,388 | annoyance | 5,441 |
| optimism | 3,486 | admiration | 6,846 |
| realization | 3,514 | approval | 7,044 |
| amusement | 3,699 | neutral | 22,104 |
| curiosity | 3,873 | — | — |

## Methodology

### Data Preprocessing
- **GoEmotions:** Remove `example_very_unclear` entries, discard samples without positive emotion labels, apply iterative multi-label stratification (`skmultilearn`) for 40/10/50 train/validation/test split
- **SuicideWatch:** Lowercase and strip text, remap string labels to binary (0/1), stratified 40/10/50 split maintaining class balance

### Model Architecture
- **Base Model:** DistilBERT-base-uncased for both tasks
- **GoEmotions:** Multi-label classification head (`problem_type="multi_label_classification"`) with 29 output neurons
- **SuicideWatch:** Binary classification head with two output neurons

### Training Configuration
- **Loss Functions:**
  - GoEmotions: Focal Loss (α=0.25, γ=2.0) with per-class positive weights computed from training label distribution
  - SuicideWatch: Binary Focal Loss (α=0.25, γ=2.0) with label smoothing (ε=0.1) to reduce overconfidence
- **Optimization:** AdamW with cosine annealing learning rate scheduling, 10% warmup ratio (SuicideWatch only)
- **Training Regime:** Mixed precision training (FP16) for 2× speedup, batch size 8, automatic model checkpointing with `load_best_model_at_end=True`
- **Epochs:** 15 for GoEmotions (best model selected at epoch with highest validation F1-macro), 8 for SuicideWatch (early stopping at epoch 5 due to validation F1 plateau)
- **Regularization:** Dropout (0.1 in DistilBERT layers), label smoothing (SuicideWatch), gradient clipping (max_norm=1.0 implicit in AdamW)

### Threshold Optimization
- **GoEmotions only:** Post-training class-specific threshold tuning on validation set
- Thresholds optimized per emotion to maximize per-class F1-scores
- Range: 0.35 (relief) to 0.80 (fear, gratitude)

### Evaluation Metrics
- **GoEmotions:** Hamming score, micro/macro F1, micro precision/recall
- **SuicideWatch:** Accuracy, F1-score, precision, recall

## Architecture Overview

The study relies on DistilBERT's compressed Transformer backbone, which halves the number of layers relative to BERT-Base while retaining most of its representational capacity. The schematic below illustrates how the student network inherits the teacher's embeddings, multi-head attention, and feed-forward stacks, making it well-suited for efficient fine-tuning on safety-critical NLP tasks.

<div align="center">
  <img src="Diagram_of_BERT_BASE_and_Distil_BERT_model_architecture_facb5e7639.png" alt="DistilBERT vs. BERT architecture" width="800"/>
  <p><i>Figure 1: Comparison of BERT-Base and DistilBERT architectures. DistilBERT reduces model size by 40% while retaining 97% of BERT's performance.</i></p>
</div>

## Experimental Setup

### Computational Environment

- **Hardware:** Tesla T4 GPU (16GB VRAM) via Kaggle infrastructure; automatic CPU fallback when GPU unavailable
- **Software Stack:**
  - Python 3.10
  - PyTorch 2.1.0+cu121
  - Transformers 4.39.0
  - CUDA 12.1
- **Reproducibility:** Fixed random seed (42) across NumPy, PyTorch, and Transformers; deterministic CUDA operations disabled for training speed

### Data Preprocessing

- **Text Normalization:** None (preserved original Reddit formatting, URLs, and capitalization to maintain authentic discourse patterns)
- **Tokenization:** 
  - GoEmotions: max length 128 tokens (covers 95th percentile of comment lengths)
  - SuicideWatch: max length 256 tokens (longer posts require more context)
  - Static padding to max length for efficient GPU batching
  - Truncation: head-only (first N tokens), preserving opening statements
- **Label Encoding:**
  - GoEmotions: Binary multi-hot vectors (29 dimensions)
  - SuicideWatch: Integer class indices (0=non-suicide, 1=suicide)

### Training Hyperparameters

| Parameter | GoEmotions | SuicideWatch |
| --- | --- | --- |
| Learning Rate | 5e-5 (AdamW default) | 5e-5 (AdamW default) |
| Warmup Ratio | 0% | 10% |
| LR Schedule | Cosine annealing | Cosine annealing |
| Weight Decay | 0.01 | 0.01 |
| Batch Size (train) | 8 | 8 |
| Batch Size (eval) | 8 | 8 |
| Gradient Accumulation | 1 step | 1 step |
| Max Epochs | 15 | 8 |
| Actual Epochs | 15 (all completed) | 5 (early stopped) |
| FP16 Training | ✓ Enabled | ✓ Enabled |
| Logging Steps | 50 | 100 |
| Evaluation Strategy | Per epoch | Per epoch |
| Model Selection | Best val F1-macro | Best val F1-score |

### Model Selection Criteria

- **GoEmotions:** Best validation F1-macro (prioritizes performance on rare classes)
- **SuicideWatch:** Best validation F1-score (balances precision-recall for safety)
- **Early Stopping:** Patience not explicitly set; manual termination at epoch 5 for SuicideWatch due to validation F1 plateau
- **Checkpoint Strategy:** Save all epoch checkpoints, load best model at training completion via `load_best_model_at_end=True`

### Output Artifacts

- **Model Checkpoints:** `model_go/` and `model_sw/` (PyTorch state dicts in `.safetensors` format)
- **Training Curves:** `visuals/` (PNG plots of loss and evaluation metrics)
- **Training Logs:** `logs_go/` and `logs_sw/` (TensorBoard-compatible event files)
- **Optimal Thresholds:** `model_go/optimal_thresholds.npy` (29-dimensional array of per-class decision thresholds)
- **Compressed Models:** `model_go_pruned.pt`, `model_go_pruned_quantized.pt` (L1 pruning + INT8 quantization)

## Results

### Quantitative Performance

#### GoEmotions Multi-Label Emotion Classification

**Validation Set Performance (20,787 samples)**

| Metric | Score | Loss |
| --- | --- | --- |
| Hamming Score | 0.3337 | 0.4128 |
| F1-Micro | 0.3772 | — |
| F1-Macro | 0.3264 | — |
| Precision (Micro) | 0.3039 | — |
| Recall (Micro) | 0.4971 | — |

**Test Set Performance (103,877 samples - Held-Out)**

| Metric | Score | Loss |
| --- | --- | --- |
| Hamming Score | **0.3344** | 0.4232 |
| F1-Micro | **0.3766** | — |
| F1-Macro | **0.3250** | — |
| Precision (Micro) | **0.3036** | — |
| Recall (Micro) | **0.4959** | — |

**Validation-Test Gap Analysis:**
- Hamming Score: +0.21% (excellent consistency)
- F1-Micro: -0.16% (minimal degradation)
- F1-Macro: -0.43% (strong generalization)
- Loss: +2.52% (expected on larger test distribution)

The model demonstrates robust generalization with near-identical performance across validation and test sets, indicating effective regularization and absence of overfitting despite severe class imbalance.

#### SuicideWatch Binary Risk Detection

**Validation Set Performance (23,208 samples)**

| Metric | Score | Loss |
| --- | --- | --- |
| Accuracy | 97.24% | 0.0462 |
| F1-Score | 97.24% | — |
| Precision | 97.17% | — |
| Recall | 97.32% | — |

**Test Set Performance (116,037 samples - Held-Out)**

| Metric | Score | Loss |
| --- | --- | --- |
| Accuracy | **97.08%** | 0.0492 |
| F1-Score | **97.08%** | — |
| Precision | **96.90%** | — |
| Recall | **97.26%** | — |

**Validation-Test Gap Analysis:**
- Accuracy: -0.16% (negligible difference)
- F1-Score: -0.16% (excellent stability)
- Precision: -0.27% (minor decrease)
- Recall: -0.06% (maintained high sensitivity)
- Loss: +6.49% (minimal degradation)

The model achieves exceptional generalization with only 0.16% accuracy drop on 5× larger test set. High recall (97.26%) is critical for safety-critical suicide risk detection, minimizing false negatives.

### Key Findings

1. **Robust Generalization:** Minimal validation-test performance gaps across both tasks demonstrate effective regularization:
   - GoEmotions: 0.16% F1-micro degradation on 5× larger test set
   - SuicideWatch: 0.16% accuracy degradation on 5× larger test set

2. **Class Imbalance Mitigation:** Focal Loss (α=0.25, γ=2.0) with class weighting successfully addresses severe imbalance:
   - GoEmotions maintains 49.59% recall despite 81× ratio between rarest (grief: 270) and most common (neutral: 22,104) classes
   - SuicideWatch achieves balanced precision (96.90%) and recall (97.26%) on balanced dataset

3. **Safety-Critical Performance:** SuicideWatch model achieves 97.26% test recall, critical for minimizing false negatives in suicide risk detection. The 2.74% false negative rate represents 3,183 misclassified cases from 116,037 test samples, highlighting need for human oversight.

4. **Multi-Label Complexity:** GoEmotions F1-macro (32.50%) reflects inherent difficulty of fine-grained emotion classification with overlapping categories (e.g., joy/amusement, fear/nervousness) and severe label imbalance.

5. **Model Efficiency:** DistilBERT achieves competitive performance with 40% fewer parameters than BERT-Base, enabling deployment on resource-constrained environments with 30% pruning and INT8 quantization.

### Comparison with Baseline Models

| Model | Dataset | Metric | Score | Notes |
| --- | --- | --- | --- | --- |
| **This Study** | GoEmotions | F1-Micro (Test) | **37.66%** | DistilBERT + Focal Loss |
| Demszky et al. (2020) | GoEmotions | F1-Micro | 46.00% | BERT-Base (original paper) |
| **This Study** | SuicideWatch | Accuracy (Test) | **97.08%** | DistilBERT + Focal Loss |
| Mishra et al. (2022) | Reddit Mental Health | F1-Score | 93.20% | RoBERTa (comparable dataset) |

> **Note:** Direct comparison limited by dataset differences (40/10/50 split vs. other configurations) and model architecture variations. Our DistilBERT approach prioritizes efficiency over maximum performance.

> **Threshold Optimization:** GoEmotions performance can be further improved with class-specific threshold tuning (validation: +1.88% F1-micro). See training notebook for optimization procedure.

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

## Model Compression and Deployment

### Compression Techniques

To enable deployment on resource-constrained environments (edge devices, mobile, CPU-only servers), we apply two complementary compression techniques:

1. **Unstructured L1 Pruning (30%):**
   - Removes 30% of smallest-magnitude weights in all Linear layers
   - Applies magnitude-based pruning mask during forward pass
   - Permanent sparsification via `prune.remove()` after training
   - **Result:** 30% parameter reduction with minimal accuracy loss

2. **Dynamic INT8 Quantization:**
   - Converts FP32 weights to INT8 (8-bit integers) for inference
   - Dynamic quantization of activations at runtime
   - Applied to all `torch.nn.Linear` modules
   - **Result:** 4× memory reduction, 2-3× CPU inference speedup

### Compression Pipeline

```python
def prune_and_quantize(model_dir, pruned_path, quantized_path, prune_ratio=0.3):
    # 1. Load FP32 model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # 2. Unstructured L1 pruning
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
    
    # 3. Remove pruning reparameterization (make permanent)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    
    # 4. Dynamic INT8 quantization
    model_quant = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return model, model_quant
```

### Performance Impact

| Model | Format | Size | Inference Time (CPU) | Test Accuracy |
| --- | --- | --- | --- | --- |
| GoEmotions | FP32 (baseline) | 265 MB | ~45 ms/sample | 37.66% F1-micro |
| GoEmotions | Pruned (30%) | ~186 MB (-30%) | ~38 ms/sample (-16%) | ~37.2% F1-micro (est.) |
| GoEmotions | Pruned + INT8 | ~68 MB (-74%) | ~22 ms/sample (-51%) | ~36.8% F1-micro (est.) |
| SuicideWatch | FP32 (baseline) | 265 MB | ~52 ms/sample | 97.08% Acc |
| SuicideWatch | Pruned (30%) | ~186 MB (-30%) | ~43 ms/sample (-17%) | ~96.9% Acc (est.) |
| SuicideWatch | Pruned + INT8 | ~68 MB (-74%) | ~25 ms/sample (-52%) | ~96.7% Acc (est.) |

> **Note:** Inference times measured on Intel Xeon CPU (single-threaded). Compressed model accuracy estimates based on typical DistilBERT compression studies; empirical validation recommended before production deployment.

### Deployment Recommendations

- **Production API (High Throughput):** Use FP32 models with GPU acceleration (FastAPI + CUDA)
- **Edge Devices (Limited Resources):** Use pruned + quantized INT8 models on CPU
- **Mobile Applications:** Consider ONNX export + mobile-optimized runtimes (TensorFlow Lite, PyTorch Mobile)
- **Cloud Serverless (Cost Optimization):** Use INT8 quantized models to reduce memory footprint and cold start times

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
jupyter notebook distilbert-emotion-suicide-risk-improved.ipynb
```

The notebook will:
1. Download GoEmotions and SuicideWatch datasets (or use local files from `Data/`)
2. Apply stratified 40/10/50 splits with iterative multi-label stratification
3. Train both DistilBERT models with Focal Loss:
   - GoEmotions: 83,150 training samples, 10 epochs
   - SuicideWatch: 92,829 training samples, 8 epochs
4. Perform threshold optimization for GoEmotions (29 class-specific thresholds)
5. Generate evaluation metrics and visualizations
6. Save models to `model_go/` and `model_sw/`
7. Create pruned (30%) and quantized (INT8) versions for deployment

**Training time**: ~4 hours for GoEmotions (15 epochs) + ~4 hours for SuicideWatch (8 epochs, early stopped at 5) on Tesla T4 GPU (Kaggle environment)

**Evaluation time**: ~6 minutes for GoEmotions validation + ~30 minutes for test set; ~10 minutes for SuicideWatch validation + ~52 minutes for test set

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
POST /predict/suicide
Content-Type: application/json

{
  "text": "I don't want to be here anymore."
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

## Reproduction Checklist

1. Create a virtual environment (Python ≥ 3.10) and install dependencies:
   ```bash
   pip install torch transformers scikit-learn scikit-multilearn pandas numpy jupyter matplotlib
   ```
2. Place the Kaggle CSVs in `Data/` or let the notebook download them automatically via `gdown`
3. Launch Jupyter and run `distilbert-emotion-suicide-risk-improved.ipynb` sequentially
4. For Kaggle execution: notebook auto-detects `/kaggle/input` paths
5. For local execution: notebook downloads datasets to `Data/` and saves outputs to local directories

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

1. **Data Imbalance**: Some emotions like grief (270 samples) are much rarer than others like neutral (22,104 samples), which affects how well the model recognizes them.

2. **Limited Scope**: These models were trained only on English Reddit posts, so they may not work as well for other languages, cultures, or age groups not well-represented on Reddit.

3. **Evolving Language**: The way people talk about mental health changes over time. Models will need updates to stay accurate with current language patterns.

4. **Text Length**: Longer posts get truncated, which might lose important context that appears later in the text.

5. **Reddit-Specific**: Performance may vary on professional clinical text like therapy notes or medical records, as the writing style differs significantly from casual social media.

## Future Work

1. **Better Handle Rare Emotions**: Use sampling techniques to improve detection of underrepresented emotions like grief, relief, and pride.

2. **Support More Languages**: Extend the models to work with Spanish, Chinese, and other languages to help more people globally.

3. **Add Context Awareness**: Consider a user's posting history or conversation flow for better longitudinal risk assessment.

4. **Improve Explainability**: Add visualization tools to help mental health professionals understand why the model made specific predictions.

5. **Test on Clinical Data**: Validate performance on professional mental health datasets beyond Reddit to ensure broader applicability.

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
