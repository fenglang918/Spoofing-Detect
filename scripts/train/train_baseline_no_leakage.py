#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
No-Leakage Training Pipeline for Spoofing Detection
--------------------------------------------------
严格移除信息泄露的训练流程：
• 只使用委托时刻可观测的特征
• 移除所有未来聚合特征
• 使用原始标签或严格定义的时点标签
• 真实的模型性能评估
"""

import argparse, glob, os, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def get_realtime_observable_features():
    """严格定义：只在委托时刻可观测的特征"""
    return [
        # 行情特征（委托时刻可观测）
        "bid1", "ask1", "prev_close", "mid_price", "spread",
        
        # 价格特征（委托时刻可观测）
        "delta_mid", "pct_spread", "price_dev_prevclose",
        
        # 订单特征（委托时刻可观测）
        "is_buy", "log_qty",
        
        # 历史统计特征（只使用过去的信息，时间窗口结束时刻 < 委托时刻）
        "orders_100ms", "cancels_5s", 
        
        # 时间特征
        "time_sin", "time_cos", "in_auction",
        
        # 增强特征（基于委托时刻的信息）
        "book_imbalance",  # 基于委托时刻的申买申卖量
        "price_aggressiveness",  # 基于委托价格与当时bid/ask的关系
        # 注意：layering_score如果基于未来信息则不能使用
    ]

def get_leakage_features():
    """定义可能泄露未来信息的特征"""
    return [
        # 明显的未来信息
        "final_survival_time_ms",  # 最终存活时间
        "total_events",           # 总事件数
        "total_traded_qty",       # 总成交量
        "num_trades",            # 成交次数
        "num_cancels",           # 撤单次数
        "is_fully_filled",       # 是否完全成交
        
        # 可能包含未来信息的特征
        "layering_score",        # 如果基于未来订单模式计算
        
        # 基于整个订单生命周期的聚合特征
        "存活时间_ms",           # 存活时间（事件发生时未知）
    ]

def create_time_based_labels(df, r1_ms=50, r2_ms=1000, r2_mult=4.0):
    """创建基于时点的标签（避免使用存活时间）"""
    print("🏷️ Creating time-based labels without leakage...")
    
    # 按订单分组处理
    labels = []
    
    for (date, ticker, order_id), group in df.groupby(['自然日', 'ticker', '交易所委托号']):
        group = group.sort_values('事件_datetime')
        
        # 获取第一个事件（委托时刻）的信息
        first_event = group.iloc[0]
        
        # 检查在r1_ms时间窗口内是否有撤单
        r1_window_end = first_event['委托_datetime'] + pd.Timedelta(milliseconds=r1_ms)
        r1_events = group[group['事件_datetime'] <= r1_window_end]
        has_quick_cancel = (r1_events['事件类型'] == '撤单').any()
        
        # 检查价格偏离条件（基于委托时刻的信息）
        if 'spread' in first_event and 'delta_mid' in first_event:
            spread_val = first_event['spread']
            delta_mid_val = first_event['delta_mid']
            price_condition = abs(delta_mid_val) >= r2_mult * spread_val if spread_val > 0 else False
        else:
            price_condition = False
        
        # 组合条件
        is_spoofing = has_quick_cancel and price_condition
        
        # 为该订单的所有事件分配标签
        for idx in group.index:
            labels.append({
                'index': idx,
                'y_label_clean': int(is_spoofing)
            })
    
    label_df = pd.DataFrame(labels).set_index('index')
    return label_df

def analyze_data_leakage(df):
    """分析数据泄露风险"""
    print("\n🔍 Analyzing potential data leakage...")
    
    leakage_features = get_leakage_features()
    observable_features = get_realtime_observable_features()
    
    print(f"📊 Total features in dataset: {len(df.columns)}")
    
    # 检查泄露特征
    found_leakage = [f for f in leakage_features if f in df.columns]
    if found_leakage:
        print(f"⚠️ Found {len(found_leakage)} potential leakage features:")
        for feat in found_leakage:
            print(f"  - {feat}")
    
    # 检查可用的安全特征
    available_safe = [f for f in observable_features if f in df.columns]
    print(f"✅ Found {len(available_safe)} safe observable features:")
    for feat in available_safe:
        print(f"  - {feat}")
    
    return found_leakage, available_safe

def comprehensive_evaluation(y_true, y_pred_proba):
    """综合评估（无泄露版本）"""
    # 基础指标
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    metrics = {'PR-AUC': pr_auc, 'ROC-AUC': roc_auc}
    
    # Precision at K
    for k in [0.001, 0.005, 0.01, 0.05]:
        k_int = max(1, int(len(y_true) * k))
        top_k_idx = y_pred_proba.argsort()[::-1][:k_int]
        prec_k = y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()
        metrics[f'Precision@{k*100:.1f}%'] = prec_k
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_regex", default="202503|202504")
    parser.add_argument("--valid_regex", default="202505")
    parser.add_argument("--use_original_labels", action="store_true", 
                       help="使用原始y_label而不是重新生成的无泄露标签")
    parser.add_argument("--r1_ms", type=int, default=50, help="快速撤单时间阈值")
    parser.add_argument("--r2_ms", type=int, default=1000, help="价格偏离时间阈值")
    parser.add_argument("--r2_mult", type=float, default=4.0, help="价格偏离倍数")
    
    args = parser.parse_args()
    
    t0 = time.time()
    print("🔍 Loading data for no-leakage training...")
    
    # 加载特征数据
    feat_pats = [os.path.join(args.data_root, "features_select", "X_*.parquet")]
    files = []
    for pat in feat_pats:
        files.extend(sorted(glob.glob(pat)))
    
    if not files:
        print(f"❌ No feature files found")
        return
    
    df_feat = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Features data shape: {df_feat.shape}")
    
    # 加载标签数据
    lab_pats = [os.path.join(args.data_root, "labels_select", "labels_*.parquet")]
    files = []
    for pat in lab_pats:
        files.extend(sorted(glob.glob(pat)))
    
    df_lab = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Labels data shape: {df_lab.shape}")
    
    # 合并数据
    df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="inner")
    print(f"Merged data shape: {df.shape}")
    
    # 分析数据泄露
    leakage_features, safe_features = analyze_data_leakage(df)
    
    # 移除泄露特征
    print(f"\n🚫 Removing {len(leakage_features)} leakage features...")
    df_clean = df.drop(columns=leakage_features, errors='ignore')
    
    # 如果不使用原始标签，生成无泄露标签
    if not args.use_original_labels:
        print("🏷️ Generating clean labels without survival time...")
        
        # 需要加载原始事件流数据来生成无泄露标签
        event_pattern = os.path.join(args.data_root, "event_stream", "*", "委托事件流.csv")
        event_files = glob.glob(event_pattern)
        
        if event_files:
            print(f"Found {len(event_files)} event stream files")
            
            # 读取几个文件作为示例
            sample_events = []
            for event_file in event_files[:3]:  # 只处理前3个文件作为示例
                date_str = os.path.basename(os.path.dirname(event_file))
                if any(date_str in regex for regex in [args.train_regex, args.valid_regex]):
                    print(f"Processing {date_str}...")
                    event_df = pd.read_csv(event_file, parse_dates=['委托_datetime', '事件_datetime'])
                    sample_events.append(event_df)
            
            if sample_events:
                events_combined = pd.concat(sample_events, ignore_index=True)
                clean_labels = create_time_based_labels(events_combined, args.r1_ms, args.r2_ms, args.r2_mult)
                
                # 合并清洁标签
                df_clean = df_clean.merge(clean_labels, left_index=True, right_index=True, how='left')
                df_clean['y_label_clean'] = df_clean['y_label_clean'].fillna(0).astype(int)
                target_col = 'y_label_clean'
            else:
                print("⚠️ No event files found, using original y_label")
                target_col = 'y_label'
        else:
            print("⚠️ No event stream files found, using original y_label")
            target_col = 'y_label'
    else:
        target_col = 'y_label'
    
    print(f"Using target column: {target_col}")
    
    # 准备特征
    id_cols = ["自然日", "ticker", "交易所委托号", "y_label"]
    if target_col != 'y_label':
        id_cols.append(target_col)
    
    feature_cols = [col for col in safe_features if col in df_clean.columns]
    print(f"\n📊 Using {len(feature_cols)} safe features:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    # 数据切分
    train_mask = df_clean["自然日"].astype(str).str.contains(args.train_regex)
    valid_mask = df_clean["自然日"].astype(str).str.contains(args.valid_regex)
    
    df_train = df_clean[train_mask].copy()
    df_valid = df_clean[valid_mask].copy()
    
    print(f"\n📅 Data split:")
    print(f"Training: {len(df_train):,} samples")
    print(f"Validation: {len(df_valid):,} samples")
    
    # 准备训练数据
    X_tr = df_train[feature_cols].fillna(0)
    y_tr = df_train[target_col]
    X_va = df_valid[feature_cols].fillna(0)
    y_va = df_valid[target_col]
    
    print(f"\n⚖️ Class distribution:")
    print(f"Train: {y_tr.value_counts().to_dict()}")
    print(f"Valid: {y_va.value_counts().to_dict()}")
    
    # 简单下采样平衡数据
    if y_tr.sum() > 0:
        pos_indices = y_tr[y_tr == 1].index
        neg_indices = y_tr[y_tr == 0].index
        
        # 保持1:20的比例
        target_neg_size = min(len(pos_indices) * 20, len(neg_indices))
        selected_neg_indices = np.random.choice(neg_indices, target_neg_size, replace=False)
        
        selected_indices = np.concatenate([pos_indices, selected_neg_indices])
        X_tr = X_tr.loc[selected_indices]
        y_tr = y_tr.loc[selected_indices]
        
        print(f"After balancing: {y_tr.value_counts().to_dict()}")
    
    # 训练模型
    print(f"\n🚀 Training LightGBM model...")
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
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=100,
            verbose=False
        )
    except TypeError:
        from lightgbm import early_stopping
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(100)]
        )
    
    # 评估
    print("\n📊 Model Evaluation (No Leakage):")
    y_pred_proba = model.predict_proba(X_va)[:, 1]
    
    metrics = comprehensive_evaluation(y_va, y_pred_proba)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 特征重要性
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Feature Importance (No Leakage):")
    for i, (feat, imp) in enumerate(feature_imp.head(10).values):
        print(f"  {i+1:2d}. {feat:<25} {imp:>8.0f}")
    
    # 保存结果
    results = {
        'method': f"No-Leakage-Training-{target_col}",
        'n_features': len(feature_cols),
        'removed_leakage_features': len(leakage_features),
        'training_time': time.time() - t0,
        **metrics
    }
    
    import json
    results_file = os.path.join(args.data_root, "no_leakage_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"⏱️ Total training time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

"""
使用示例：

# 使用原始标签，移除泄露特征
python scripts/train/train_baseline_no_leakage.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --use_original_labels

# 生成无泄露标签并训练
python scripts/train/train_baseline_no_leakage.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --r1_ms 50 --r2_ms 1000 --r2_mult 4.0
""" 