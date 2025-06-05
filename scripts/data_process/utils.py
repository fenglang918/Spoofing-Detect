#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py
────────────────────────────────────────
工具函数和配置模块
"""

import pandas as pd
import numpy as np
import os
from rich.console import Console

# Configure console
console = Console()

# ──────────────────────────── Config ────────────────────────────
STRICT = True  # True = 缺列立即报错；False = 自动填 NaN

# 设置 Polars 线程数
os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())

# ──────────────────────────── Utility Functions ────────────────────────────
def require_cols(df: pd.DataFrame, cols: list[str]):
    """严格模式：保证必须列全部存在，否则抛错"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要行情列：{missing}")

def apply_feature_whitelist(df_pd: pd.DataFrame) -> pd.DataFrame:
    """应用特征白名单，移除信息泄露特征"""
    
    # 黑名单：必须移除的信息泄露特征
    blacklist_patterns = [
        'is_cancel$',  # 不是is_cancel_event，而是原来的is_cancel
        'total_events', 'total_traded_qty', 'num_trades', 'num_cancels',
        'final_survival_time_ms', 'is_fully_filled',
        'flag_R1', 'flag_R2'  # 旧版标签规则的中间变量
    ]
    
    # 检测常数列
    const_cols = []
    for col in df_pd.columns:
        if df_pd[col].nunique() <= 1:
            const_cols.append(col)
    
    # 合并要删除的列
    cols_to_drop = const_cols.copy()
    for pattern in blacklist_patterns:
        import re
        matching_cols = [col for col in df_pd.columns if re.search(pattern, col)]
        cols_to_drop.extend(matching_cols)
    
    # 去重
    cols_to_drop = list(set(cols_to_drop))
    
    if cols_to_drop:
        console.print(f"  🚫 Removing {len(cols_to_drop)} blacklisted/constant features: {cols_to_drop}")
        df_pd = df_pd.drop(columns=cols_to_drop, errors='ignore')
    
    # 白名单：确保保留的实时特征
    whitelist_features = [
        '自然日', 'ticker', '交易所委托号',  # 主键
        'bid1', 'ask1', 'mid_price', 'spread', 'prev_close', 'bid_vol1', 'ask_vol1',  # 盘口快照
        'log_qty', 'is_buy',  # 订单静态特征
        'orders_100ms', 'orders_1s', 'cancels_100ms', 'cancels_1s', 'cancels_5s',  # 短期历史窗口
        'cancel_ratio_100ms', 'cancel_ratio_1s', 'trades_1s',  # 撤单率和成交统计
        'time_sin', 'time_cos', 'in_auction',  # 时间周期特征
        'delta_mid', 'pct_spread', 'price_dev_prevclose_bps',  # 价格相关（已修正）
        'book_imbalance', 'price_aggressiveness', 'cluster_score',  # 衍生稳定指标
        'z_survival', 'price_momentum_100ms', 'spread_change', 'order_density',  # 新增特征
        'is_cancel_event',  # 事件标记（当前时刻可观测）
        'layering_score'  # 如果存在且为实时版本
    ]
    
    # 标签相关列（如果存在）
    label_patterns = ['y_label', 'spoofing', 'manipulation', 'liquidity', 'layering', 'cancel_impact']
    for col in df_pd.columns:
        if any(pattern in col for pattern in label_patterns):
            whitelist_features.append(col)
    
    # 保留白名单中存在的列
    available_features = [col for col in whitelist_features if col in df_pd.columns]
    
    console.print(f"  ✅ Keeping {len(available_features)} whitelisted features")
    console.print(f"  📊 Feature categories:")
    console.print(f"    • Market snapshot: {len([c for c in available_features if c in ['bid1','ask1','mid_price','spread','bid_vol1','ask_vol1','prev_close']])}")
    console.print(f"    • Order static: {len([c for c in available_features if c in ['log_qty','is_buy','委托价格']])}")
    console.print(f"    • Rolling windows: {len([c for c in available_features if 'orders_' in c or 'cancels_' in c or 'trades_' in c])}")
    console.print(f"    • Derived indicators: {len([c for c in available_features if c in ['book_imbalance','price_aggressiveness','cluster_score']])}")
    
    return df_pd[available_features] 