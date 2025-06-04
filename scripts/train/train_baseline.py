#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM Training Pipeline
---------------------------------
• 读取 features_select/ 与 labels_select/ 下的 Parquet
• 通过日期正则切分 train / valid
• is_unbalance / scale_pos_weight 二选一处理类别失衡
• 支持 GPU (device_type="gpu")，自动回退 CPU
"""

import argparse, glob, os, re, sys, time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

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

# ---------- Main -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="根目录，包含 features_select/ labels_select/")
    parser.add_argument("--train_regex", default="202503|202504", help="训练集日期正则")
    parser.add_argument("--valid_regex", default="202505", help="验证集日期正则")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--is_unbalance", type=bool, default=True)
    parser.add_argument("--scale_pos_weight", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--n_estimators", type=int, default=3000)
    parser.add_argument("--early_stop", type=int, default=200)
    args = parser.parse_args()

    t0 = time.time()
    feat_pats = [os.path.join(args.data_root, "features_select", "X_*.parquet")]
    lab_pats  = [os.path.join(args.data_root, "labels_select", "labels_*.parquet")]
    df_feat = load_parquets(feat_pats)
    df_lab  = load_parquets(lab_pats)
    df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="inner", validate="one_to_one")

    # ---- Train / Valid split
    df_train, df_valid = split_by_date(df, args.train_regex)
    df_valid, df_drop  = split_by_date(df_valid, args.valid_regex)  # valid=正则匹配
    if df_valid.empty:
        sys.exit("❌ 找不到验证集日期，请检查 --valid_regex")
    
    # Drop identifier columns that shouldn't be features
    id_cols = ["自然日", "ticker", "交易所委托号", "y_label"]
    feature_cols = [col for col in df_train.columns if col not in id_cols]
    
    X_tr = df_train[feature_cols]
    y_tr = df_train["y_label"]
    X_va = df_valid[feature_cols]
    y_va = df_valid["y_label"]

    cat_cols = []  # No categorical features since we dropped identifier columns

    # ---- LightGBM params
    params = dict(
        device_type=args.device,
        boosting_type="gbdt",
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=5,
        min_sum_hessian_in_leaf=1e-3,
        random_state=42,
        categorical_feature=cat_cols,
        early_stopping_rounds=args.early_stop,
    )
    if args.is_unbalance:
        params["is_unbalance"] = True
    if args.scale_pos_weight:
        params["scale_pos_weight"] = args.scale_pos_weight

    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",  # PR-AUC
    )

    # ---- Metrics
    y_prob = clf.predict_proba(X_va)[:, 1]
    pr_auc = average_precision_score(y_va, y_prob)
    prec_k = precision_at_k(y_va.reset_index(drop=True), pd.Series(y_prob), 0.001)

    print(f"\n🏁 Training done in {time.time()-t0:.1f}s")
    print(f"Best iteration: {clf.best_iteration_}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print(f"Precision@Top0.1%: {prec_k:.4f}")

    # ---- Save model
    model_path = os.path.join(args.data_root, "model_lgbm.txt")
    clf.booster_.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

"""
python scripts/train/train_baseline.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --device "cpu" \
  --is_unbalance True \
  --learning_rate 0.05 \
  --num_leaves 63 \
  --n_estimators 3000 \
  --early_stop 200

"""