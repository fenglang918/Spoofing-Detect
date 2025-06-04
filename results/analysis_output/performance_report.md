# Spoofing Detection Model Performance Report

Generated on: 2025-06-05 02:12:15

## Experiments Overview

### ensemble_results
- PR-AUC: 1.000000
- ROC-AUC: 1.000000
- Features: N/A
- Training time: N/A
- Positive ratio: N/A

### enhanced_results
- PR-AUC: 1.000000
- ROC-AUC: 1.000000
- Features: N/A
- Training time: N/A
- Positive ratio: N/A

### baseline_results
- PR-AUC: 0.030194
- ROC-AUC: N/A
- Features: N/A
- Training time: N/A
- Positive ratio: N/A

### optimized_results
- PR-AUC: 1.000000
- ROC-AUC: 1.000000
- Features: N/A
- Training time: N/A
- Positive ratio: N/A

## Performance Comparison

|                   | method                                                       |   PR-AUC |   Precision@0.1% |   ROC-AUC |   success |
|:------------------|:-------------------------------------------------------------|---------:|-----------------:|----------:|----------:|
| ensemble_results  | Model ensemble + Undersample_composite_spoofing              | 1        |         0.750831 |         1 |         1 |
| enhanced_results  | Enhanced features + Undersample_composite_spoofing           | 1        |         0.750831 |         1 |         1 |
| baseline_results  | LightGBM + Undersample                                       | 0.030194 |         0.0709   |       nan |       nan |
| optimized_results | Hyperparameter optimization + Undersample_composite_spoofing | 1        |         0.750831 |         1 |         1 |


## Recommendations

- 🏆 Best performing model: ensemble_results (PR-AUC: 1.0000)
- 💡 Significant improvement achieved: 3211.9% gain from optimization

