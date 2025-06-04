#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training with Enhanced Labels (No Data Leakage)
----------------------------------------------
使用无泄露增强标签进行训练，对比不同标签策略的效果
"""

import argparse, glob, os, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

def get_safe_features():
    """获取安全的（无数据泄露）特征列表"""
    return [
        # 行情特征（委托时刻可观测）
        "bid1", "ask1", "prev_close", "mid_price", "spread",
        
        # 价格特征（委托时刻可观测）
        "delta_mid", "pct_spread", "price_dev_prevclose",
        
        # 订单特征（委托时刻可观测）
        "is_buy", "log_qty",
        
        # 历史统计特征（只使用过去的信息）
        "orders_100ms", "cancels_5s", 
        
        # 时间特征
        "time_sin", "time_cos", "in_auction",
        
        # 增强特征（基于委托时刻的信息）
        "book_imbalance", "price_aggressiveness"
    ]

def comprehensive_evaluation(y_true, y_pred_proba, label_name=""):
    """综合评估函数"""
    metrics = {}
    
    # 基础指标
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    metrics['PR-AUC'] = pr_auc
    metrics['ROC-AUC'] = roc_auc
    
    # Precision at K
    for k in [0.001, 0.005, 0.01, 0.05]:
        k_int = max(1, int(len(y_true) * k))
        top_k_idx = y_pred_proba.argsort()[::-1][:k_int]
        prec_k = y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()
        metrics[f'Precision@{k*100:.1f}%'] = prec_k
    
    # 打印结果
    print(f"\n📊 {label_name} Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return metrics

def train_with_label_strategy(X_train, y_train, X_valid, y_valid, strategy_name, balance_ratio=20):
    """使用特定标签策略训练模型"""
    print(f"\n🚀 Training with {strategy_name}...")
    
    # 数据平衡
    if y_train.sum() > 0:
        pos_indices = y_train[y_train == 1].index
        neg_indices = y_train[y_train == 0].index
        
        if len(neg_indices) > 0:
            target_neg_size = min(len(pos_indices) * balance_ratio, len(neg_indices))
            selected_neg_indices = np.random.choice(neg_indices, target_neg_size, replace=False)
            
            selected_indices = np.concatenate([pos_indices, selected_neg_indices])
            X_train_balanced = X_train.loc[selected_indices]
            y_train_balanced = y_train.loc[selected_indices]
            
            print(f"  Balanced data: {y_train_balanced.value_counts().to_dict()}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        print(f"  ⚠️ No positive samples found for {strategy_name}")
        return None, None
    
    # 训练模型
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='average_precision',
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=10,
        reg_lambda=10,
        random_state=42,
        verbose=-1
    )
    
    # 训练
    try:
        model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=False
        )
    except TypeError:
        from lightgbm import early_stopping
        model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_valid, y_valid)],
            callbacks=[early_stopping(100)]
        )
    
    # 预测
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    
    # 评估
    metrics = comprehensive_evaluation(y_valid, y_pred_proba, strategy_name)
    
    return model, metrics

def compare_label_strategies(df_train, df_valid, feature_cols):
    """对比不同标签策略的效果"""
    print("\n🔬 Comparing Label Strategies")
    print("=" * 60)
    
    # 准备特征
    X_train = df_train[feature_cols].fillna(0)
    X_valid = df_valid[feature_cols].fillna(0)
    
    results = {}
    
    # 策略1: 原始标签
    if 'y_label' in df_train.columns:
        y_train_orig = df_train['y_label']
        y_valid_orig = df_valid['y_label']
        
        model_orig, metrics_orig = train_with_label_strategy(
            X_train, y_train_orig, X_valid, y_valid_orig, 
            "Original Labels (y_label)"
        )
        if metrics_orig:
            results['original'] = metrics_orig
    
    # 策略2: 增强标签 - 宽松版
    if 'enhanced_spoofing_liberal' in df_train.columns:
        y_train_lib = df_train['enhanced_spoofing_liberal']
        y_valid_lib = df_valid['enhanced_spoofing_liberal']
        
        model_lib, metrics_lib = train_with_label_strategy(
            X_train, y_train_lib, X_valid, y_valid_lib,
            "Enhanced Liberal Labels"
        )
        if metrics_lib:
            results['enhanced_liberal'] = metrics_lib
    
    # 策略3: 增强标签 - 中等版
    if 'enhanced_spoofing_moderate' in df_train.columns:
        y_train_mod = df_train['enhanced_spoofing_moderate'] 
        y_valid_mod = df_valid['enhanced_spoofing_moderate']
        
        model_mod, metrics_mod = train_with_label_strategy(
            X_train, y_train_mod, X_valid, y_valid_mod,
            "Enhanced Moderate Labels"
        )
        if metrics_mod:
            results['enhanced_moderate'] = metrics_mod
    
    # 策略4: 增强标签 - 严格版
    if 'enhanced_spoofing_strict' in df_train.columns:
        y_train_strict = df_train['enhanced_spoofing_strict']
        y_valid_strict = df_valid['enhanced_spoofing_strict']
        
        model_strict, metrics_strict = train_with_label_strategy(
            X_train, y_train_strict, X_valid, y_valid_strict,
            "Enhanced Strict Labels"
        )
        if metrics_strict:
            results['enhanced_strict'] = metrics_strict
    
    # 策略5: 综合增强标签
    if 'enhanced_combined' in df_train.columns:
        y_train_comb = df_train['enhanced_combined']
        y_valid_comb = df_valid['enhanced_combined']
        
        model_comb, metrics_comb = train_with_label_strategy(
            X_train, y_train_comb, X_valid, y_valid_comb,
            "Enhanced Combined Labels"
        )
        if metrics_comb:
            results['enhanced_combined'] = metrics_comb
    
    return results

def create_comparison_report(results, output_dir):
    """创建对比报告"""
    print("\n📊 Label Strategy Comparison Report")
    print("=" * 80)
    
    # 创建对比表格
    comparison_df = pd.DataFrame(results).T
    
    # 打印主要指标对比
    key_metrics = ['PR-AUC', 'ROC-AUC', 'Precision@0.1%', 'Precision@1.0%']
    print(f"\n🎯 Key Metrics Comparison:")
    print(comparison_df[key_metrics].round(6))
    
    # 找出最佳策略
    best_pr_auc = comparison_df['PR-AUC'].idxmax()
    best_prec_01 = comparison_df['Precision@0.1%'].idxmax()
    
    print(f"\n🏆 Best Strategies:")
    print(f"  Best PR-AUC: {best_pr_auc} ({comparison_df.loc[best_pr_auc, 'PR-AUC']:.6f})")
    print(f"  Best Precision@0.1%: {best_prec_01} ({comparison_df.loc[best_prec_01, 'Precision@0.1%']:.6f})")
    
    # 计算改进程度
    if 'original' in results and 'enhanced_combined' in results:
        orig_pr_auc = results['original']['PR-AUC']
        enh_pr_auc = results['enhanced_combined']['PR-AUC']
        improvement = (enh_pr_auc - orig_pr_auc) / orig_pr_auc * 100
        
        print(f"\n📈 Enhancement Impact:")
        print(f"  Original PR-AUC: {orig_pr_auc:.6f}")
        print(f"  Enhanced PR-AUC: {enh_pr_auc:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    # 保存详细报告
    output_file = output_dir / "label_strategy_comparison.csv"
    comparison_df.to_csv(output_file, float_format='%.8f')
    print(f"\n💾 Detailed comparison saved to: {output_file}")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--enhanced_dir", help="增强标签目录（默认为data_root/labels_enhanced_enhanced）")
    parser.add_argument("--train_regex", default="202503|202504", help="训练数据日期正则")
    parser.add_argument("--valid_regex", default="202505", help="验证数据日期正则")
    parser.add_argument("--results_dir", help="结果保存目录（默认为data_root/enhanced_label_results）")
    
    args = parser.parse_args()
    
    print("🚀 Training with Enhanced Labels (No Data Leakage)")
    print("=" * 70)
    
    data_root = Path(args.data_root)
    
    # 设置目录
    if args.enhanced_dir:
        enhanced_dir = Path(args.enhanced_dir)
    else:
        enhanced_dir = data_root / "labels_enhanced_enhanced"
    
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = data_root / "enhanced_label_results"
    
    results_dir.mkdir(exist_ok=True)
    
    print(f"📁 Data root: {data_root}")
    print(f"📁 Enhanced labels: {enhanced_dir}")
    print(f"📁 Results dir: {results_dir}")
    
    # 检查增强标签是否存在
    if not enhanced_dir.exists():
        print(f"❌ Enhanced labels directory not found: {enhanced_dir}")
        print("Please run enhanced labeling first:")
        print(f"python scripts/data_process/enhanced_labeling_no_leakage.py --data_root {data_root}")
        return
    
    # 加载特征数据
    feat_files = list((data_root / "features_select").glob("X_*.parquet"))
    if not feat_files:
        print("❌ No feature files found")
        return
    
    df_features = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    print(f"📊 Features shape: {df_features.shape}")
    
    # 加载原始标签
    label_files = list((data_root / "labels_select").glob("labels_*.parquet"))
    df_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
    print(f"📊 Original labels shape: {df_labels.shape}")
    
    # 加载增强标签
    enhanced_files = list(enhanced_dir.glob("enhanced_labels_*.parquet"))
    if not enhanced_files:
        print(f"❌ No enhanced label files found in {enhanced_dir}")
        return
    
    df_enhanced = pd.concat([pd.read_parquet(f) for f in enhanced_files], ignore_index=True)
    print(f"📊 Enhanced labels shape: {df_enhanced.shape}")
    
    # 合并所有数据
    df = df_features.merge(df_labels, on=['自然日', 'ticker', '交易所委托号'], how='inner')
    df = df.merge(df_enhanced.drop('y_label', axis=1, errors='ignore'), on=['自然日', 'ticker', '交易所委托号'], how='inner')
    print(f"📊 Merged data shape: {df.shape}")
    
    # 数据切分
    train_mask = df["自然日"].astype(str).str.contains(args.train_regex)
    valid_mask = df["自然日"].astype(str).str.contains(args.valid_regex)
    
    df_train = df[train_mask].copy()
    df_valid = df[valid_mask].copy()
    
    print(f"\n📅 Data split:")
    print(f"Training: {len(df_train):,} samples")
    print(f"Validation: {len(df_valid):,} samples")
    
    # 准备特征
    feature_cols = get_safe_features()
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"\n📊 Using {len(available_features)} safe features:")
    for i, feat in enumerate(available_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # 对比不同标签策略
    results = compare_label_strategies(df_train, df_valid, available_features)
    
    # 创建对比报告
    if results:
        comparison_df = create_comparison_report(results, results_dir)
        
        # 保存完整结果
        import json
        results_file = results_dir / "enhanced_label_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Full results saved to: {results_file}")
    else:
        print("❌ No valid results obtained")

if __name__ == "__main__":
    main()

"""
使用示例：

# 1. 首先生成增强标签
python scripts/data_process/enhanced_labeling_no_leakage.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data"

# 2. 使用增强标签进行训练对比
python scripts/train/train_with_enhanced_labels.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505"
""" 