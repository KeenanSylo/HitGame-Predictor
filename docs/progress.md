# HitGame Predictor – Model Progress Log

## Overview
This project predicts whether a Steam game becomes a “hit” using metadata from SteamSpy and the Steam Storefront.  
Model: **XGBoost Classifier**  
Dataset size: **478 games**

---

## Model Performance Summary

| Version | Description | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|----------|--------------|----------|--------|------------|---------|----|-----------|
| **v0.1 (2025-10-30)** | Baseline model | 0.796 | — | 0.50 | 0.08 | 0.13 | 0.86 |
| **v0.2 (2025-10-31)** | Weighted + PR-tuned model | **0.855** | **0.469** | 0.43 | **0.69** | **0.53** | 0.83 |

**Improvement:** Recall ↑ from 0.08 → 0.69, F1 ↑ from 0.13 → 0.53, ROC-AUC ↑ from 0.796 → 0.855.

---

## v0.1 – Baseline
- Label: hit if owners_est ≥ 1M  
- Model: XGBoost (300 trees, learning_rate=0.1, max_depth=5)  
- Result: high accuracy but poor recall for hits.

---

## v0.2 – Weighted & PR-Tuned
- Added `scale_pos_weight` to balance classes.  
- Used PR-AUC as eval metric with early stopping.  
- Tuned decision threshold for better F1.  
- Result: much better recall and overall balance.

---

## Next Steps
- Add new features (tags, price ratio, description length).  
- Try cross-validation for more stable metrics.  
- Possibly refine hit-label definition by percentiles.
