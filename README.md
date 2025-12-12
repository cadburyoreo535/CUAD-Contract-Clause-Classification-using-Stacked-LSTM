# Legal Contract Clause Classification using Stacked LSTM

## CCS 248 – Artificial Neural Networks Final Project

---

## Problem Statement

**Automated classification of legal contract clauses** (e.g., parties, governing law, effective/expiration dates).

---

## Solution
- **Model**: 2-layer BiLSTM with attention pooling and dropout
- **Tokenization**: Custom tokenizer (10k vocab), max length 11 tokens (85th percentile of filtered snippets)
- **Balancing**: Class weights + optional sampler; optional synonym augmentation for low-support classes
- **Framework**: PyTorch

### Architecture
```
Embedding (200)
BiLSTM(128×2) → Dropout 0.25
BiLSTM(96×2)  → Attention pooling
Linear → 7 classes
```

---

## Dataset
- Source: CUAD v1 `label_group_xlsx/` sheets (flattened columns → rows of `context, clause_type`)
- Filtering: keep classes with ≥5 samples (TOP_N=20 cap in notebook); optional synonym augmentation to lift low-support labels
- Tokenization: custom vocab (~2.6k+ words observed), 85th-percentile length → max pad length 11
- Notes: `openpyxl` required for XLSX loading; `nltk`+WordNet used only if augmentation is enabled

---

## Training & Tuning
- Split: 70/15/15 stratified (train/val/test)
- Optimizers in notebook (run5 sweep): Adam/RMSprop (lr 5e-4–1e-3), batch 64, epochs 5–10, patience 6, ReduceLROnPlateau (0.5, 2), grad clip 1.0
- Class handling: weights + optional weighted sampler; optional synonym augmentation via WordNet

### Results (runs 1–5)
- Run2 (CSV pipeline, tuned): best test acc ≈ 74.3% in run2 (RMSprop lr=8e-4, batch 64); metrics in `experiment_results_run2.csv`; artifacts in `artifacts_run2/`; models in `trained_models_run2/`.
- Run3 (tokenizer/artifacts saved): models/artifacts present (`trained_models_run3/`, `artifacts_run3/`).
- Run4 (current XLSX pipeline): models/artifacts present (`trained_models_run4/`, `artifacts_run4/`).
- Run5 (extra sweep scaffolded): artifacts folder exists (`trained_models_run5/`,`artifacts_run5/`).

---

## Baseline
- TF-IDF + Logistic Regression (sanity check on label quality).

---

## How to Run
1) Install deps (XLSX + optional augmentation):
```bash
pip install torch numpy pandas scikit-learn h5py openpyxl nltk
```
If using augmentation, download NLTK data inside the notebook (`nltk.download('wordnet'); nltk.download('omw-1.4')`).

2) Open and run `Untitled-6.ipynb` end-to-end. It will:
  - Load XLSX snippets from `CUAD_v1/label_group_xlsx/`
  - Clean, optionally augment low-support classes, tokenize, split, and train the BiLSTM+attention
  - Save models to `trained_models_run5/` (current notebook paths)
  - Save metrics to `experiment_results_run2.csv` (legacy name—update in notebook if you want per-run CSVs)
  - Save artifacts to `artifacts_run5/`

---

## Files
```
ANNFINAL/
├── Untitled-6.ipynb          # Main PyTorch notebook
├── w.ipynb                   # Keras reference notebook
├── README.md
├── experiment_results.csv       # Early run1 metrics (low accuracy)
├── experiment_results_run2.csv  # Run2 metrics (best ~74.3% test); also reused by later runs unless renamed
├── trained_models_run2/         # Run2 models
├── artifacts_run2/              # Run2 tokenizer/labels/confusion/reports
├── trained_models_run3/         # Run3 models
├── artifacts_run3/              # Run3 tokenizer/labels/confusion/reports
├── trained_models_run4/         # Run4 (XLSX) models
├── artifacts_run4/              # Run4 tokenizer/labels/confusion/reports
├── trained_models_run5/         # Current notebook output path
├── artifacts_run5/              # Current notebook output path
├── trained_models/              # Prior Keras/PyTorch models
├── artifacts/                   # Prior artifacts
└── CUAD_v1/
    ├── CUAD_v1.json
    ├── CUAD_v1_README.txt
    ├── master_clauses.csv
    └── full_contract_txt/
```

---

## References
- CUAD: Hendrycks et al., 2021, arXiv:2103.06268 (https://www.atticusprojectai.org/cuad)

---

## Author
CCS 248 – Artificial Neural Networks (Dec 2025)
