# Legal Contract Clause Classification using Stacked LSTM

## CCS 248 – Artificial Neural Networks Final Project

---

## Problem Statement

**Automated classification of legal contract clauses** (e.g., parties, governing law, effective/expiration dates).

---

## Solution
- **Model**: 2-layer BiLSTM with attention pooling and dropout
- **Tokenization**: Custom tokenizer (10k vocab), max length 11 tokens (85th percentile of filtered snippets)
- **Balancing**: Class weights + optional sampler
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
- Filtering: keep classes with ≥5 samples (no TOP_N cap recommended); optional synonym augmentation to lift low-support labels
- Tokenization: custom vocab (~2.6k+ words observed), 85th-percentile length → max pad length 11
- Note: `openpyxl` is required for XLSX loading; `nltk`+WordNet used only if augmentation is enabled

---

## Training & Tuning
- Split: 70/15/15 stratified (train/val/test)
- Optimizers in notebook: Adam/RMSprop (5–8e-4), batch 64, epochs 5 (adjust upward for full runs), patience 6, ReduceLROnPlateau (0.5, 2), grad clip 1.0
- Class handling: weights + optional weighted sampler; optional synonym augmentation via WordNet

### Results (runs 1–5)
- Run2 (CSV pipeline, tuned): best test acc ≈ 74.3% (RMSprop lr=8e-4, batch 64); metrics in `experiment_results_run2.csv`; artifacts in `artifacts_run2/`; models in `trained_models_run2/`.
- Run3 (tokenizer/artifacts saved): models/artifacts present (`trained_models_run3/`, `artifacts_run3/`); metrics CSV not recorded.
- Run4 (current XLSX pipeline): models/artifacts present (`trained_models_run4/`, `artifacts_run4/`); results CSV still named `experiment_results_run2.csv` in the notebook—rename if you rerun.
- Run5 (extra sweep scaffolded): artifacts folder exists (`artifacts_run5/`); metrics CSV not recorded.

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
  - Save models to `trained_models_run4/` (current run tag)
  - Save metrics to `experiment_results_run2.csv` (legacy name—adjust if desired)
  - Save artifacts to `artifacts_run4/`

---

## Files
```
ANNFINAL/
├── Untitled-6.ipynb          # Main PyTorch notebook
├── w.ipynb                   # Keras reference notebook
├── README.md
├── experiment_results_run2.csv  # Legacy results file name (used by notebook)
├── trained_models_run4/         # Saved PyTorch checkpoints (.pt, .h5) for current run tag
├── artifacts_run4/              # Tokenizer, labels, confusion matrix, reports (current)
├── trained_models/           # Prior run models
├── artifacts/               # Prior run artifacts
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
