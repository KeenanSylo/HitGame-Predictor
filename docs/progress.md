# HitGame Predictor – Model Progress Log

## Overview
The goal of this project is to predict whether a Steam game will become a "hit" based on metadata from SteamSpy and the Steam Storefront.  
Model: **XGBoost Classifier**  
Dataset size (after merge): **478 games**

---

## 2025-10-30 — Initial Baseline Model

### Dataset
- Total samples: **478**
- Positive (hit): **67**
- Negative (non-hit): **411**
- Labeling rule: owners_est ≥ 10^6 (1 million)
- Free-to-play titles: removed
- Train/validation split: 80/20 stratified

### Model Setup
| Parameter | Value |
|------------|--------|
| Algorithm | XGBoost (binary:logistic) |
| Version | 3.1.1 |
| n_estimators | 300 |
| learning_rate | 0.1 |
| max_depth | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| random_state | 42 |

### Evaluation (baseline threshold = 0.5)

| Metric | Value |
|---------|--------|
| ROC-AUC | **0.796** |
| Accuracy | **0.86** |
| Precision (class 1) | **0.50** |
| Recall (class 1) | **0.08** |
| F1-score (class 1) | **0.13** |
| Samples tested | 96 |

### Confusion Matrix (approx.)
|          | Pred 0 | Pred 1 |
|-----------|---------|--------|
| **True 0** | 82 | 1 |
| **True 1** | 12 | 1 |

### Interpretation
- The model easily identifies non-hits (class 0) but rarely predicts hits (class 1).
- High accuracy is misleading due to class imbalance.
- ROC-AUC is promising, suggesting there is signal in the data.
- Next step: improve **recall** for class 1 (hits) by:
  1. Adding `scale_pos_weight` to handle imbalance.
  2. Using `aucpr` (PR-AUC) as eval metric.
  3. Applying early stopping and dynamic probability threshold.
  4. Exploring percentile-based labeling and release-year filtering.

---

## Planned Next Experiments

| Date | Experiment | Goal |
|------|-------------|------|
| TBD | Add `scale_pos_weight` (neg/pos) | Improve hit recall |
| TBD | Tune threshold by PR curve | Better F1 balance |
| TBD | Add early stopping & validation split | Prevent overfit |
| TBD | Try percentile-based label | Robust label balance |
| TBD | Feature importance review | Understand drivers of “hit” |

---

## Notes
- Model file saved to: `data/models/xgb_hit_predictor.pkl`
- Dataset file: `data/features/base_dataset.csv`
- Next improvement commit: **v0.2 — Weighted & PR-tuned model**
