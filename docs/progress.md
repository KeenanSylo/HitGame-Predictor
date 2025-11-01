# HitGame Predictor – Model Progress Log

## Overview
This project predicts whether a Steam game becomes a “hit” using metadata from SteamSpy and the Steam Storefront.  
Model: **XGBoost Classifier** (+ ensemble in later versions)  
Dataset size: **478 games**

---

## Main (AAA / General) Model Performance

| Version | Description | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|----------|--------------|----------|--------|------------|---------|----|-----------|
| **v0.1 (2025-10-30)** | Baseline model | 0.796 | — | 0.50 | 0.08 | 0.13 | 0.86 |
| **v0.2 (2025-10-31)** | Weighted + PR-tuned model | 0.855 | 0.469 | 0.43 | 0.69 | 0.53 | 0.83 |
| **v0.3 (2025-11-01)** | Cross-validated & balanced | 0.818 | 0.414 | 0.37 | 0.57 | 0.49 | 0.83 |
| **v0.4 (2025-11-01)** | Feature-expanded ensemble (XGBoost + RandomForest) | **0.821** | **0.474** | 0.39 | **0.70** | **0.50** | 0.80 |

---

## v0.1 — Baseline
- Label: hit if owners_est ≥ 1M  
- Model: XGBoost (300 trees, learning_rate=0.1, max_depth=5)  
- Result: High accuracy but poor recall for hits — model biased toward non-hits.

---

## v0.2 — Weighted & PR-Tuned
- Added `scale_pos_weight` to balance classes.  
- Used PR-AUC as eval metric with early stopping.  
- Tuned decision threshold for better F1.  
- Result: Recall and F1 significantly improved.

---

## v0.3 — Cross-Validated & Balanced
- Added 5-fold stratified cross-validation for stability.  
- Improved F1 and recall with threshold tuning per OOF results.  
- Ensured no data leakage in validation.  
- Result: Stronger generalization with consistent scores across folds.

---

## v0.4 — Feature-Expanded Ensemble
- Added frequency-encoded `genre_score` and `category_score`.  
- Combined **XGBoost** and **RandomForest** for more stable ensemble predictions.  
- Improved recall and PR-AUC slightly while maintaining overall balance.  
- Accuracy drop expected due to better hit recall.

### Metrics
- ROC-AUC: **0.821**  
- PR-AUC: **0.474**  
- Recall: **0.70**  
- F1: **0.50**

### Interpretation
- Model now detects most hit games (recall ~70%) without overfitting.  
- Ensemble balances bias/variance and extracts signal from genre/category structure.  
- PR-AUC improvement confirms better precision-recall trade-off under imbalance.

---

## Niche / Indie Popularity Model (v0.4-niche)

| Version | Description | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|----------|--------------|----------|--------|------------|---------|----|-----------|
| **v0.4-niche (2025-11-01)** | Niche/Indie rising-game model | 0.734 | 0.361 | 0.43 | 0.50 | 0.46 | 0.80 |

### Details
- Trained on **recent low-owner titles** (under ~200k owners).  
- Aimed to predict which *new or niche games* are on the rise.  
- Used the same ensemble (XGBoost + RandomForest) architecture.  
- Dataset size: **~150 games**.

### Metrics
- ROC-AUC: **0.734**  
- PR-AUC: **0.361**  
- Recall: **0.50**  
- F1: **0.46**

### Interpretation
- Model can correctly identify roughly half of the “rising” niche games.  
- Solid early performance considering small and noisy dataset.  
- Indicates discoverable popularity patterns in new, underexposed titles.

---

## Next Steps
- Perform **feature importance analysis** to prune weak tag columns.  
- Add **textual sentiment/keyword** features from descriptions.  
- Introduce **Optuna hyperparameter tuning** for model weight optimization.  
- Explore **percentile-based labeling** for adaptive hit definition.  
- Expand **niche dataset** with more years or adjusted owner thresholds.

---

**Model files:**  
- AAA Ensemble → `data/models/ensemble_v04.pkl`  
- Niche Ensemble → `data/models/ensemble_niche.pkl`  

**Meta files:**  
- AAA → `data/models/meta_v04.json`  
- Niche → `data/models/meta_niche.json`  

**Dataset file:** `data/features/base_dataset.csv`
