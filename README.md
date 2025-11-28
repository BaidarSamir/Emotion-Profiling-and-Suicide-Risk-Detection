# DistilBERT-Based Emotion Profiling and Suicide Risk Detection

## Abstract

We investigate whether a compact Transformer (DistilBERT) can simultaneously support nuanced emotion profiling and binary suicide risk detection for text-based mental-health support systems. Using the GoEmotions corpus and a curated SuicideWatch dataset, we construct task-specific heads, apply class-weighted loss functions, and evaluate both validation and test performance. The study further quantifies label imbalance, monitors convergence behavior, and applies pruning plus dynamic quantization to produce deployable INT8 checkpoints.

## Repository Overview

```
.
├── Data/
│   ├── go_emotions_dataset.csv
│   └── Suicide_Detection.csv
└── distilbert-emotion-suicide-risk.ipynb
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
- **Evaluation Metrics:** Hamming score plus micro/macro F1 (emotions) and accuracy/precision/recall/F1 (SuicideWatch). Sigmoid threshold defaults to 0.5 for all emotions

## Experimental Setup

- Hardware: single GPU runtime (Kaggle/Colab-style environment; adjust to local CUDA device if available).
- Tokenization: max length 128 (GoEmotions) and 256 (SuicideWatch), padding to fixed length.
- Optimizer/TrainingArguments: batch size 8, 10 epochs for GoEmotions, 5 epochs for SuicideWatch, evaluation at each epoch, logging every 50 steps.
- Outputs: checkpoints and logs stored in `/kaggle/working/...`; update the notebook paths if running locally.

## Results

| Task | Validation metrics | Test metrics |
| --- | --- | --- |
| GoEmotions | `F1_micro = 0.362`, `F1_macro = 0.295`, `hamming = 0.317` | `F1_micro = 0.357`, `F1_macro = 0.315`, `hamming = 0.314` |
| SuicideWatch | `accuracy = 0.951`, `F1 = 0.952`, `precision = 0.949`, `recall = 0.954` | `accuracy = 0.961`, `F1 = 0.961`, `precision = 0.958`, `recall = 0.964` |

### Training Dynamics

GoEmotions fine-tuning (`batch_size = 8`, 10 epochs):

| Epoch | Train Loss | Val Loss | Hamming | F1 Micro | F1 Macro | Precision Micro | Recall Micro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.045 | 0.961 | 0.162 | 0.227 | 0.212 | 0.134 | 0.738 |
| 2 | 0.832 | 0.958 | 0.195 | 0.252 | 0.234 | 0.153 | 0.709 |
| 3 | 0.724 | 1.157 | 0.230 | 0.300 | 0.260 | 0.192 | 0.692 |
| 4 | 0.532 | 1.290 | 0.237 | 0.303 | 0.269 | 0.198 | 0.643 |
| 5 | 0.392 | 1.524 | 0.256 | 0.322 | 0.281 | 0.220 | 0.600 |
| 6 | 0.324 | 1.839 | 0.269 | 0.330 | 0.280 | 0.235 | 0.552 |
| 7 | 0.271 | 2.031 | 0.292 | 0.345 | 0.290 | 0.255 | 0.537 |
| 8 | 0.198 | 2.379 | 0.298 | 0.348 | 0.289 | 0.265 | 0.507 |
| 9 | 0.182 | 2.489 | 0.312 | 0.358 | 0.292 | 0.282 | 0.490 |
| 10 | 0.151 | 2.583 | 0.317 | 0.362 | 0.295 | 0.289 | 0.485 |

SuicideWatch fine-tuning (5 epochs):

| Epoch | Train Loss | Val Loss | Accuracy | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.121 | 0.164 | 0.951 | 0.952 | 0.938 | 0.966 |
| 2 | 0.094 | 0.173 | 0.955 | 0.955 | 0.960 | 0.950 |
| 3 | 0.013 | 0.249 | 0.952 | 0.953 | 0.942 | 0.964 |
| 4 | 0.001 | 0.312 | 0.952 | 0.953 | 0.949 | 0.956 |
| 5 | 0.0004 | 0.316 | 0.951 | 0.952 | 0.949 | 0.954 |

## Model Compression and Deployment

Both fine-tuned checkpoints undergo 30 % unstructured L1 pruning on all linear layers, followed by dynamic quantization (`torch.quantization.quantize_dynamic`) to obtain INT8 models. This reduces on-disk size and inference latency, making the models suitable for edge-serving backends.

### Example Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = torch.load("model_sw_pruned_quantized.pt")
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
3. Launch Jupyter (`jupyter notebook`) and run `distilbert-emotion-suicide-risk.ipynb` sequentially.
4. Update `output_dir` and checkpoint paths if running outside Kaggle to avoid writing to `/kaggle/working`.

## Ethical Considerations

These models analyze text about self-harm and should only augment, never replace, trained professionals. Deployments must incorporate human-in-the-loop review, clear escalation policies, and guardrails against misuse. Maintain user privacy and obtain all necessary approvals before processing sensitive content.

## Future Directions

1. Increase GoEmotions coverage (larger training subset or class-balanced sampling) to improve macro-level recall.
2. Integrate experiment tracking (Weights & Biases, MLflow) for hyper-parameter sweeps.
3. Wrap the quantized checkpoints in a FastAPI/Gradio service with confidence calibration and explanation tooling.
