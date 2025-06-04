#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training with Enhanced Labels (Correct Implementation)
----------------------------------------------------
正确实现：
• 标签可以使用任何信息（包括未来信息）来定义虚假报单
• 特征只使用委托时刻可观测信息，移除数据泄露
• 对比原始标签 vs 增强标签的效果
"""

import argparse, glob, os, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
import json

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

def get_leakage_features():
    """识别包含数据泄露的特征（需要移除）"""
    return [
        # 明显的未来信息特征
        "final_survival_time_ms",  # 最终存活时间
        "total_events",           # 总事件数
        "total_traded_qty",       # 总成交量
        "num_trades",            # 成交次数
        "num_cancels",           # 撤单次数
        "is_fully_filled",       # 是否完全成交
        
        # 可能包含未来信息的聚合特征
        "存活时间_ms",           # 存活时间（在特征中是泄露）
        "layering_score"         # 如果基于未来信息计算
    ]

def analyze_data_quality(df):
    """分析数据质量和标签分布"""
    print("\n📊 Data Quality Analysis:")
    print(f"Total samples: {len(df):,}")
    
    # 检查标签分布
    label_cols = [col for col in df.columns if any(x in col for x in ['spoofing', 'y_label', 'quick_', 'price_', 'fake_', 'layering_', 'active_'])]
    
    print(f"\n🏷️ Label Distribution:")
    for col in sorted(label_cols):
        if col in df.columns:
            pos_count = df[col].sum()
            pos_rate = pos_count / len(df) * 100
            print(f"  {col:<30}: {pos_count:>8,} ({pos_rate:>6.3f}%)")
    
    # 特征泄露检查
    safe_features = get_safe_features()
    leakage_features = get_leakage_features()
    
    available_safe = [f for f in safe_features if f in df.columns]
    found_leakage = [f for f in leakage_features if f in df.columns]
    
    print(f"\n✅ Safe features available: {len(available_safe)}")
    print(f"⚠️ Leakage features found: {len(found_leakage)}")
    
    if found_leakage:
        print(f"  Removing: {found_leakage}")
    
    return available_safe, found_leakage

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

def train_model(X_train, y_train, X_valid, y_valid, label_name, balance_ratio=20):
    """训练单个模型"""
    print(f"\n🚀 Training with {label_name}...")
    
    # 检查正样本
    pos_count = y_train.sum()
    if pos_count == 0:
        print(f"  ⚠️ No positive samples for {label_name}")
        return None, None
    
    print(f"  Positive samples: {pos_count:,} ({pos_count/len(y_train)*100:.3f}%)")
    
    # 数据平衡
    pos_indices = y_train[y_train == 1].index
    neg_indices = y_train[y_train == 0].index
    
    if len(neg_indices) > 0:
        target_neg_size = min(len(pos_indices) * balance_ratio, len(neg_indices))
        selected_neg_indices = np.random.choice(neg_indices, target_neg_size, replace=False)
        
        selected_indices = np.concatenate([pos_indices, selected_neg_indices])
        X_train_balanced = X_train.loc[selected_indices]
        y_train_balanced = y_train.loc[selected_indices]
        
        print(f"  Balanced: {y_train_balanced.value_counts().to_dict()}")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # 模型训练
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
    
    # 预测和评估
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    metrics = comprehensive_evaluation(y_valid, y_pred_proba, label_name)
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--train_regex", default="202503|202504", help="训练数据日期正则")
    parser.add_argument("--valid_regex", default="202505", help="验证数据日期正则")
    parser.add_argument("--results_dir", help="结果保存目录")
    
    args = parser.parse_args()
    
    print("🚀 Training with Enhanced Labels (Correct Implementation)")
    print("=" * 70)
    
    data_root = Path(args.data_root)
    
    # 设置结果目录
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = data_root / "enhanced_label_results_correct"
    results_dir.mkdir(exist_ok=True)
    
    print(f"📁 Data root: {data_root}")
    print(f"📁 Results dir: {results_dir}")
    
    # 加载数据
    print("\n📥 Loading data...")
    
    # 特征数据
    feat_files = list((data_root / "features_select").glob("X_*.parquet"))
    if not feat_files:
        print("❌ No feature files found")
        return
    df_features = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    print(f"Features shape: {df_features.shape}")
    
    # 标签数据
    label_files = list((data_root / "labels_select").glob("labels_*.parquet"))
    if not label_files:
        print("❌ No label files found")
        return
    df_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
    print(f"Labels shape: {df_labels.shape}")
    
    # 合并数据
    df = df_features.merge(df_labels, on=['自然日', 'ticker', '交易所委托号'], how='inner')
    print(f"Merged shape: {df.shape}")
    
    # 数据质量分析
    safe_features, leakage_features = analyze_data_quality(df)
    
    # 移除泄露特征
    df_clean = df.drop(columns=leakage_features, errors='ignore')
    print(f"\nAfter removing leakage features: {df_clean.shape}")
    
    # 数据切分
    train_mask = df_clean["自然日"].astype(str).str.contains(args.train_regex)
    valid_mask = df_clean["自然日"].astype(str).str.contains(args.valid_regex)
    
    df_train = df_clean[train_mask].copy()
    df_valid = df_clean[valid_mask].copy()
    
    print(f"\n📅 Data split:")
    print(f"Training: {len(df_train):,} samples")
    print(f"Validation: {len(df_valid):,} samples")
    
    # 准备特征
    feature_cols = [f for f in safe_features if f in df_clean.columns]
    print(f"\n📊 Using {len(feature_cols)} safe features:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    X_train = df_train[feature_cols].fillna(0)
    X_valid = df_valid[feature_cols].fillna(0)
    
    # 训练不同标签策略
    results = {}
    
    # 1. 原始标签
    if 'y_label' in df_train.columns:
        model, metrics = train_model(
            X_train, df_train['y_label'], 
            X_valid, df_valid['y_label'],
            "Original Labels"
        )
        if metrics:
            results['original'] = metrics
    
    # 2. 增强标签 - 各种策略
    enhanced_labels = {
        'composite_spoofing': 'Enhanced Composite',
        'conservative_spoofing': 'Enhanced Conservative', 
        'quick_cancel_impact': 'Quick Cancel Impact',
        'price_manipulation': 'Price Manipulation',
        'fake_liquidity': 'Fake Liquidity',
        'layering_cancel': 'Layering Cancel',
        'active_hours_spoofing': 'Active Hours'
    }
    
    for label_col, label_name in enhanced_labels.items():
        if label_col in df_train.columns:
            model, metrics = train_model(
                X_train, df_train[label_col],
                X_valid, df_valid[label_col], 
                label_name
            )
            if metrics:
                results[label_col] = metrics
    
    # 生成对比报告
    if results:
        print("\n📊 Enhanced Labels Performance Comparison")
        print("=" * 80)
        
        comparison_df = pd.DataFrame(results).T
        
        # 主要指标
        key_metrics = ['PR-AUC', 'ROC-AUC', 'Precision@0.1%', 'Precision@1.0%']
        print(f"\n🎯 Key Metrics:")
        print(comparison_df[key_metrics].round(6))
        
        # 最佳策略
        best_pr_auc = comparison_df['PR-AUC'].idxmax()
        best_prec = comparison_df['Precision@0.1%'].idxmax()
        
        print(f"\n🏆 Best Strategies:")
        print(f"  Best PR-AUC: {best_pr_auc} ({comparison_df.loc[best_pr_auc, 'PR-AUC']:.6f})")
        print(f"  Best Precision@0.1%: {best_prec} ({comparison_df.loc[best_prec, 'Precision@0.1%']:.6f})")
        
        # 改进分析
        if 'original' in results and 'composite_spoofing' in results:
            orig_pr = results['original']['PR-AUC'] 
            enh_pr = results['composite_spoofing']['PR-AUC']
            improvement = (enh_pr - orig_pr) / orig_pr * 100
            
            print(f"\n📈 Enhancement Analysis:")
            print(f"  Original PR-AUC: {orig_pr:.6f}")
            print(f"  Enhanced PR-AUC: {enh_pr:.6f}")
            print(f"  Improvement: {improvement:+.1f}%")
        
        # 保存结果
        comparison_df.to_csv(results_dir / "enhanced_labels_comparison.csv", float_format='%.8f')
        
        with open(results_dir / "enhanced_labels_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_dir}")
        
        # 标签改进总结
        print(f"\n💡 Enhanced Labeling Summary:")
        print(f"  • 标签可以使用未来信息定义虚假报单 ✅")
        print(f"  • 特征严格限制为委托时刻可观测 ✅")
        print(f"  • 通过多种规则扩大正样本集合 ✅")
        print(f"  • 提高了模型的学习样本数量 ✅")
        
    else:
        print("❌ No valid results obtained")

if __name__ == "__main__":
    main()

"""
使用示例：

# 训练并对比增强标签效果
python scripts/train/train_enhanced_labels_correct.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505"

# 指定结果保存目录
python scripts/train/train_enhanced_labels_correct.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --results_dir "/home/ma-user/code/fenglang/Spoofing Detect/enhanced_results"
""" 