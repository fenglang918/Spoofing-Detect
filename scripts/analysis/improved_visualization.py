#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Model Visualization Analysis (English Version)
This script generates improved visualizations for model performance, 
specifically designed for imbalanced datasets.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import os

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_prediction_data(file_path='results/train_results/prediction_results_20250615_143931.json'):
    """Load prediction data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['true_labels']), np.array(data['predicted_probabilities'])

def create_improved_visualizations(y_true, y_pred_proba, output_dir='results/improved_plots'):
    """Create and save a comprehensive set of improved visualization charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Main 6-subplot figure ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Comprehensive Model Performance Analysis for Imbalanced Data', fontsize=16)

    pos_probs = y_pred_proba[y_true == 1]
    neg_probs = y_pred_proba[y_true == 0]
    
    # Subplot 1: Separated probability distribution (log scale)
    axes[0, 0].hist(neg_probs, bins=50, alpha=0.7, label=f'Negative (n={len(neg_probs):,})', color='blue', density=True)
    axes[0, 0].hist(pos_probs, bins=50, alpha=0.8, label=f'Positive (n={len(pos_probs):,})', color='red', density=True)
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('1. Probability Distribution (Log Scale)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Box plot comparison
    axes[0, 1].boxplot([neg_probs, pos_probs], labels=['Negative', 'Positive'])
    axes[0, 1].set_ylabel('Predicted Probability')
    axes[0, 1].set_title('2. Probability Distribution Box Plot')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Probability ranking scatter plot
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    sorted_probs = y_pred_proba[sorted_idx]
    sorted_labels = y_true[sorted_idx]
    n_display = min(5000, len(sorted_probs))
    colors = ['red' if label == 1 else 'blue' for label in sorted_labels[:n_display]]
    alphas = [0.8 if label == 1 else 0.3 for label in sorted_labels[:n_display]]
    axes[0, 2].scatter(range(n_display), sorted_probs[:n_display], c=colors, alpha=alphas, s=10)
    axes[0, 2].set_xlabel('Sample Rank (by Probability Desc.)')
    axes[0, 2].set_ylabel('Predicted Probability')
    axes[0, 2].set_title('3. Ranking Scatter (Red=Positive)')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Subplot 4: Multi-threshold performance curves
    thresholds = np.logspace(-6, -1, 100)
    precisions, recalls, f1_scores = [], [], []
    for thresh in thresholds:
        y_pred_binary = (y_pred_proba >= thresh).astype(int)
        tp = ((y_true == 1) & (y_pred_binary == 1)).sum()
        fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
        fn = ((y_true == 1) & (y_pred_binary == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    axes[1, 0].plot(thresholds, precisions, label='Precision', linewidth=2)
    axes[1, 0].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[1, 0].plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Performance Metrics')
    axes[1, 0].set_title('4. Multi-Threshold Performance Curves')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 5: Top-K performance visualization
    k_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    precisions_at_k, actual_thresholds = [], []
    for k in k_values:
        k_samples = max(1, int(len(y_true) * k / 100))
        top_k_idx = np.argsort(y_pred_proba)[::-1][:k_samples]
        prec_k = y_true[top_k_idx].mean()
        thresh_k = y_pred_proba[top_k_idx[-1]] if k_samples <= len(y_pred_proba) else 0
        precisions_at_k.append(prec_k)
        actual_thresholds.append(thresh_k)
    
    bars = axes[1, 1].bar(range(len(k_values)), precisions_at_k, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Top K%')
    axes[1, 1].set_ylabel('Precision@K')
    axes[1, 1].set_title('5. Top-K Precision Analysis')
    axes[1, 1].set_xticks(range(len(k_values)), [f'{k}%' for k in k_values])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Subplot 6: Adaptive threshold confusion matrix
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_f1_idx]
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2],
                xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['True Neg', 'True Pos'])
    axes[1, 2].set_title(f'6. Confusion Matrix (Thresh={optimal_threshold:.5f})')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'improved_analysis_en.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Detailed probability analysis figure ---
    fig_detail, axes_detail = plt.subplots(2, 2, figsize=(12, 10))
    fig_detail.suptitle('Detailed Probability & Curve Analysis', fontsize=16)

    # Subplot 1: Cumulative Distribution Function (CDF)
    x_neg, y_neg = ecdf(neg_probs)
    x_pos, y_pos = ecdf(pos_probs)
    axes_detail[0, 0].plot(x_neg, y_neg, label='Negative CDF', color='blue')
    axes_detail[0, 0].plot(x_pos, y_pos, label='Positive CDF', color='red')
    axes_detail[0, 0].set_xlabel('Predicted Probability')
    axes_detail[0, 0].set_ylabel('Cumulative Probability')
    axes_detail[0, 0].set_title('CDF of Probabilities')
    axes_detail[0, 0].set_xscale('log')
    axes_detail[0, 0].legend()
    axes_detail[0, 0].grid(True, alpha=0.3)

    # Subplot 2: ROC Curve (Zoomed)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    axes_detail[1, 0].plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes_detail[1, 0].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes_detail[1, 0].set_xlim(0, 0.2)
    axes_detail[1, 0].set_ylim(0, 0.6)
    axes_detail[1, 0].set_xlabel('False Positive Rate')
    axes_detail[1, 0].set_ylabel('True Positive Rate')
    axes_detail[1, 0].set_title('ROC Curve (Zoomed In)')
    axes_detail[1, 0].legend(loc='lower right')
    axes_detail[1, 0].grid(True, alpha=0.3)

    # Subplot 3: PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    axes_detail[1, 1].plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    baseline = y_true.sum() / len(y_true)
    axes_detail[1, 1].axhline(y=baseline, color='gray', linestyle='--', label=f'Random Baseline ({baseline:.3f})')
    axes_detail[1, 1].set_xlabel('Recall')
    axes_detail[1, 1].set_ylabel('Precision')
    axes_detail[1, 1].set_title('Precision-Recall Curve')
    axes_detail[1, 1].legend()
    axes_detail[1, 1].grid(True, alpha=0.3)
    
    fig_detail.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'detailed_probability_analysis_en.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_detail)
    
    print(f"✅ Improved visualization charts saved to: {output_dir}")

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y

def main():
    print("📈 Generating improved English visualizations...")
    y_true, y_pred_proba = load_prediction_data()
    create_improved_visualizations(y_true, y_pred_proba)
    print("✅ Visualization generation complete.")

if __name__ == "__main__":
    main() 