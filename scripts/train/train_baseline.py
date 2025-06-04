#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_baseline.py
──────────────────
从 features_select / labels_select 读取 Parquet，
训练 LightGBM 并在时间上独立的验证集评估 PR-AUC 与 Precision@TopK。
"""

import glob, pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_score
import re
import sys

# 检查 GPU 支持
try:
    lgb.basic._LIB.LGBM_GetLastError()
    print("LightGBM GPU 支持检查...")
    params = {'device_type': 'gpu'}
    lgb.basic._LIB.LGBM_GetLastError()
    print("✓ LightGBM GPU 支持已启用")
except Exception as e:
    print("✗ LightGBM GPU 支持未启用")
    print("请安装支持 GPU 的 LightGBM 版本:")
    print("1. 使用 pip: pip install lightgbm --install-option=--gpu")
    print("2. 或使用 conda: conda install -c conda-forge lightgbm-gpu")
    sys.exit(1)

FEAT_DIR = Path("/obs/users/fenglang/general/Spoofing Detect/data/features_select")
LBL_DIR  = Path("/obs/users/fenglang/general/Spoofing Detect/data/labels_select")

def load(month_pattern: str):
    """加载特征和标签文件，支持正则表达式匹配月份"""
    # 获取所有特征文件
    all_feat_files = glob.glob(str(FEAT_DIR / "X_*.parquet"))
    
    # 使用正则表达式过滤文件
    feat_files = [f for f in all_feat_files if re.search(f"X_2025{month_pattern}", f)]
    
    if not feat_files:
        raise ValueError(f"未找到匹配模式 '{month_pattern}' 的特征文件")
    
    lbl_files = [f.replace("features_select", "labels_select")
                   .replace("X_", "labels_") for f in feat_files]
    
    print(f"加载文件数量: {len(feat_files)}")
    print(f"文件示例: {feat_files[0]}")
    
    feats = pd.concat([pd.read_parquet(f) for f in feat_files])
    lbls = pd.concat([pd.read_parquet(f) for f in lbl_files])
    return feats.merge(lbls, on=["交易所委托号","ticker"])

# 1) 数据切分  -------------------------------------------------
train_df = load("(03|04)")   # 3月+4月
val_df   = load("05")        # 5月

X_tr = train_df.drop(columns=["交易所委托号","ticker","y_label"])
y_tr = train_df["y_label"]
X_va = val_df.drop(columns=["交易所委托号","ticker","y_label"])
y_va = val_df["y_label"]

print(f"训练集: {len(train_df):,} 样本, 正样本率: {y_tr.mean():.4%}")
print(f"验证集: {len(val_df):,} 样本, 正样本率: {y_va.mean():.4%}")

# 2) LightGBM  -------------------------------------------------
clf = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    # device_type="gpu",     
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight = y_tr.value_counts()[0] / y_tr.value_counts()[1],
    n_jobs = 4,
    random_state = 42
)

# 打印 GPU 训练信息
print("\n使用 GPU 训练 LightGBM...")
print(f"训练数据大小: {X_tr.shape}")
print(f"特征数量: {X_tr.shape[1]}")
print(f"正样本数量: {y_tr.sum():,}")
print(f"负样本数量: {len(y_tr) - y_tr.sum():,}")

clf.fit(X_tr, y_tr)

# 3) 评估  -----------------------------------------------------
proba = clf.predict_proba(X_va)[:,1]
pr_auc = average_precision_score(y_va, proba)

# Precision@Top 0.1%
k = max(1, int(0.001 * len(y_va)))
topk_idx = np.argsort(proba)[-k:]
precision_topk = precision_score(y_va.iloc[topk_idx], np.ones(k))

print(f"PR-AUC      : {pr_auc:.4f}")
print(f"Precision@{k} (≈0.1%) : {precision_topk:.4%}")
print("正样本率     :", y_va.mean().round(4))

# 4) 特征重要性  -----------------------------------------------
imp = pd.Series(clf.feature_importances_, index=X_tr.columns)\
        .sort_values(ascending=False)[:20]
print("\nTop-20 Feature Importance:")
print(imp.to_string())
