# DistilBERT-Based Emotion Profiling and Suicide Risk Detection

## Abstract

We investigate whether a compact Transformer (DistilBERT) can simultaneously support nuanced emotion profiling and binary suicide risk detection for text-based mental-health support systems. Using the GoEmotions corpus and a curated SuicideWatch dataset, we construct task-specific heads, apply class-weighted loss functions, and evaluate both validation and test performance. The study further quantifies label imbalance, monitors convergence behavior, and applies pruning plus dynamic quantization to produce deployable INT8 checkpoints.

## Repository Overview

```
.
├── Data/
│   ├── go_emotions_dataset.csv
│   └── Suicide_Detection.csv
├── emotions-and-suicide-risk-using-distilbert-model (2).ipynb
├── distilbert-emotion-suicide-risk.ipynb
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

### Architecture Overview

The study relies on DistilBERT's compressed Transformer backbone, which halves the number of layers relative to BERT-Base while retaining most of its representational capacity. The schematic below (adapted from Hugging Face) illustrates how the student network inherits the teacher's embeddings, multi-head attention, and feed-forward stacks, making it well-suited for efficient fine-tuning on safety-critical NLP tasks.

![DistilBERT vs. BERT architecture](Diagram_of_BERT_BASE_and_Distil_BERT_model_architecture_facb5e7639.png)

## Experimental Setup

- Hardware: single Tesla T4 GPU in Kaggle; the notebook auto-detects and falls back to CPU if unavailable.
- Tokenization: max length 128 (GoEmotions) and 256 (SuicideWatch) with static padding for efficient batching.
- Optimizer/TrainingArguments: batch size 8 for both tasks, GoEmotions trained for 10 epochs, SuicideWatch for 3 epochs, evaluation strategy = `epoch`, logging every 50 steps.
- Outputs: checkpoints, logs, and Matplotlib figures written to the workspace (e.g., `/kaggle/working` in Kaggle or the repo root locally).

## Results

| Task | Validation metrics | Test metrics |
| --- | --- | --- |
| GoEmotions | `hamming = 0.3209`, `F1_micro = 0.3640`, `F1_macro = 0.3156`, `precision_micro = 0.2975`, `recall_micro = 0.4687` | `hamming = 0.3209`, `F1_micro = 0.3640`, `F1_macro = 0.3156`, `precision_micro = 0.2975`, `recall_micro = 0.4687` |
| SuicideWatch | `accuracy = 0.9740`, `F1 = 0.9739`, `precision = 0.9759`, `recall = 0.9720` | `accuracy = 0.9630`, `F1 = 0.9632`, `precision = 0.9596`, `recall = 0.9667` |

### Visual Diagnostics

The notebook exports Matplotlib figures to `visuals/`, covering:

- GoEmotions training/validation loss curves plus micro-/macro-F1 and Hamming score trends.
- SuicideWatch training/validation loss and accuracy/precision/recall/F1 across epochs.
- Validation/test metric tables embedded directly in the notebook for reproducibility.

## Model Compression and Deployment

Both fine-tuned checkpoints undergo 30 % unstructured L1 pruning on every linear layer, followed by dynamic quantization (`torch.quantization.quantize_dynamic`) to obtain INT8 models. The outputs are stored alongside the original folders:

- `model_go/model_go_pruned.pt`
- `model_go/model_go_pruned_quantized.pt`
- `model_sw/model_sw_pruned.pt`
- `model_sw/model_sw_pruned_quantized.pt`

These artifacts trim disk usage and enable CPU-only inference without retraining.

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
