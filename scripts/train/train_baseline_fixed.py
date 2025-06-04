#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM Training Pipeline - DATA LEAKAGE FIXED
-------------------------------------------------
• 移除数据泄露：只使用委托时刻的实时可观测特征
• 严格时间序列验证：训练集在验证集之前
• 重新平衡数据集以获得更好的泛化性能
"""

import argparse, glob, os, re, sys, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------- Utils ----------------------------------------------------------
def load_parquets(patterns):
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise FileNotFoundError(f"No file matched {patterns}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def split_by_date(df, regex):
    mask = df["自然日"].astype(str).str.contains(regex)
    return df[mask], df[~mask]

def precision_at_k(y_true, y_score, k_ratio=0.001):
    k = max(1, int(len(y_true) * k_ratio))
    top_k_idx = y_score.argsort()[::-1][:k]
    return y_true.iloc[top_k_idx].mean()

def get_realtime_features():
    """只返回在委托时刻可观测的特征（无未来信息）"""
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
        "time_sin", "time_cos", "in_auction"
    ]

def remove_leakage_features(df):
    """移除包含未来信息的特征"""
    leakage_features = [
        # 直接的未来信息
        "存活时间_ms", "final_survival_time_ms",
        
        # 成交相关（委托时刻未知）
        "total_traded_qty", "num_trades", "is_fully_filled",
        
        # 聚合统计（包含未来事件）
        "total_events", "num_cancels",
        
        # 撤单标志（委托时刻未知是否会撤单）
        "is_cancel"
    ]
    
    available_features = [col for col in df.columns if col not in leakage_features]
    print(f"移除了 {len(leakage_features)} 个泄露特征")
    print(f"保留了 {len(available_features)} 个特征")
    
    return df[available_features]

def validate_temporal_order(df_train, df_valid):
    """验证时间序列顺序：训练集应在验证集之前"""
    train_max_date = df_train["自然日"].max()
    valid_min_date = df_valid["自然日"].min()
    
    if train_max_date >= valid_min_date:
        print(f"⚠️ 警告：时间泄露检测！训练集最大日期 ({train_max_date}) >= 验证集最小日期 ({valid_min_date})")
        return False
    else:
        print(f"✅ 时间序列验证通过：训练集 (≤{train_max_date}) → 验证集 (≥{valid_min_date})")
        return True

def balance_dataset(df, target_col="y_label", method="undersample", max_ratio=10):
    """平衡数据集以改善模型泛化"""
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    
    print(f"原始数据分布：正样本 {len(pos):,}, 负样本 {len(neg):,}")
    
    if len(neg) == 0 or len(pos) == 0:
        print("⚠️ 数据集中缺少正样本或负样本")
        return df
    
    ratio = len(neg) / len(pos)
    print(f"原始不平衡比例：{ratio:.1f}:1")
    
    if ratio > max_ratio and method == "undersample":
        # 下采样负样本
        target_neg_size = int(len(pos) * max_ratio)  # 确保是整数
        neg_sampled = neg.sample(n=min(target_neg_size, len(neg)), random_state=42)
        balanced_df = pd.concat([pos, neg_sampled], ignore_index=True)
        print(f"下采样后：正样本 {len(pos):,}, 负样本 {len(neg_sampled):,}")
    else:
        balanced_df = df
        print("保持原始分布")
    
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------- Main -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="根目录，包含 features_select/ labels_select/")
    parser.add_argument("--train_regex", default="202503|202504", help="训练集日期正则")
    parser.add_argument("--valid_regex", default="202505", help="验证集日期正则")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--balance_method", choices=["none", "undersample"], default="undersample")
    parser.add_argument("--max_imbalance_ratio", type=float, default=20.0)
    
    # LightGBM参数
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=100)
    parser.add_argument("--reg_lambda", type=float, default=10.0)
    parser.add_argument("--reg_alpha", type=float, default=10.0)
    
    args = parser.parse_args()

    t0 = time.time()
    print("🔍 Loading data...")
    
    # Load data
    feat_pats = [os.path.join(args.data_root, "features_select", "X_*.parquet")]
    lab_pats  = [os.path.join(args.data_root, "labels_select", "labels_*.parquet")]
    df_feat = load_parquets(feat_pats)
    df_lab  = load_parquets(lab_pats)
    
    print(f"特征数据形状：{df_feat.shape}")
    print(f"标签数据形状：{df_lab.shape}")
    
    # Merge datasets
    df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="inner", validate="one_to_one")
    print(f"合并后数据形状：{df.shape}")

    # Remove data leakage
    print("\n🚫 移除数据泄露特征...")
    df = remove_leakage_features(df)

    # Train / Valid split by date
    print("\n📅 按日期切分数据...")
    df_train, df_remaining = split_by_date(df, args.train_regex)
    df_valid, df_drop = split_by_date(df_remaining, args.valid_regex)
    
    if df_valid.empty:
        sys.exit("❌ 找不到验证集日期，请检查 --valid_regex")
    
    print(f"训练集：{len(df_train):,} 样本")
    print(f"验证集：{len(df_valid):,} 样本")
    
    # Validate temporal order
    print("\n⏰ 验证时间序列...")
    validate_temporal_order(df_train, df_valid)
    
    # Balance training data
    if args.balance_method != "none":
        print("\n⚖️ 平衡训练数据...")
        df_train = balance_dataset(df_train, method=args.balance_method, max_ratio=args.max_imbalance_ratio)
    
    # Prepare features
    id_cols = ["自然日", "ticker", "交易所委托号", "y_label"]
    feature_cols = [col for col in df_train.columns if col not in id_cols]
    realtime_features = get_realtime_features()
    
    # Filter to only use realtime observable features
    available_realtime = [f for f in realtime_features if f in feature_cols]
    if len(available_realtime) != len(realtime_features):
        missing = set(realtime_features) - set(available_realtime)
        print(f"⚠️ 缺少特征：{missing}")
    
    feature_cols = available_realtime
    print(f"\n📊 使用 {len(feature_cols)} 个实时可观测特征：")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    X_tr = df_train[feature_cols]
    y_tr = df_train["y_label"]
    X_va = df_valid[feature_cols]
    y_va = df_valid["y_label"]
    
    # Check for missing values
    print(f"\n🔍 数据质量检查：")
    print(f"训练集缺失值：{X_tr.isnull().sum().sum()}")
    print(f"验证集缺失值：{X_va.isnull().sum().sum()}")
    
    # Fill missing values if any
    if X_tr.isnull().sum().sum() > 0 or X_va.isnull().sum().sum() > 0:
        X_tr = X_tr.fillna(0)
        X_va = X_va.fillna(0)
        print("填充缺失值为0")

    # Print class distribution
    print(f"\n📈 类别分布：")
    print(f"训练集 - 正样本: {y_tr.sum():,} ({y_tr.mean():.4%})")
    print(f"验证集 - 正样本: {y_va.sum():,} ({y_va.mean():.4%})")

    # LightGBM params with regularization
    print(f"\n🚀 训练 LightGBM 模型...")
    params = dict(
        device_type=args.device,
        boosting_type="gbdt",
        objective="binary",
        metric="average_precision",
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_data_in_leaf=20,
        min_sum_hessian_in_leaf=1e-3,
        random_state=42,
        early_stopping_rounds=args.early_stop,
        verbose=-1
    )

    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
    )

    # Predictions and metrics
    print(f"\n📊 模型评估：")
    y_prob = clf.predict_proba(X_va)[:, 1]
    pr_auc = average_precision_score(y_va, y_prob)
    prec_k = precision_at_k(y_va.reset_index(drop=True), pd.Series(y_prob), 0.001)

    print(f"训练完成时间：{time.time()-t0:.1f}s")
    print(f"最佳迭代：{clf.best_iteration_}")
    print(f"PR-AUC：{pr_auc:.6f}")
    print(f"Precision@Top0.1%：{prec_k:.4f}")

    # Feature importance
    print(f"\n🔝 特征重要性 Top 10：")
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (feat, imp) in enumerate(feature_imp.head(10).values):
        print(f"  {i+1:2d}. {feat:<20} {imp:>8.0f}")

    # Binary classification metrics
    y_pred = (y_prob > 0.5).astype(int)
    print(f"\n📋 分类报告（阈值=0.5）：")
    print(classification_report(y_va, y_pred, target_names=['正常', '欺诈']))

    # Save model
    model_path = os.path.join(args.data_root, "model_lgbm_fixed.txt")
    clf.booster_.save_model(model_path)
    print(f"\n💾 模型已保存：{model_path}")
    
    # Save feature importance
    feature_imp.to_csv(os.path.join(args.data_root, "feature_importance.csv"), index=False)
    print(f"特征重要性已保存：{os.path.join(args.data_root, 'feature_importance.csv')}")

if __name__ == "__main__":
    main()

"""
# 修复数据泄露后的训练命令
python scripts/train/train_baseline_fixed.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --device "cpu" \
  --balance_method "undersample" \
  --max_imbalance_ratio 20.0 \
  --learning_rate 0.1 \
  --num_leaves 31 \
  --max_depth 6 \
  --n_estimators 1000 \
  --early_stop 100 \
  --reg_lambda 10.0 \
  --reg_alpha 10.0
""" 