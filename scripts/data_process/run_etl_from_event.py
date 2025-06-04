#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_etl_from_event.py
─────────────────────
1. 输入  : event_stream/<YYYYMMDD>/委托事件流.csv
2. 输出  : features_select/X_<YYYYMMDD>.parquet
           labels_select/labels_<YYYYMMDD>.parquet
3. 调用  :
      python run_etl_from_event.py \
          --root /obs/.../event_stream \
          --tickers 000989.SZ 300233.SZ \
          --r1_ms 50 --r2_ms 1000 --r2_mult 4.0 \
          --backend polars \
          --enhanced_labels  # 新增：使用增强标签
"""

import pandas as pd, numpy as np
import polars as pl
from pathlib import Path
import argparse, re, sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich import print as rprint
from concurrent.futures import ProcessPoolExecutor
import os
import warnings
warnings.filterwarnings('ignore')

# Configure console
console = Console()

# ──────────────────────────── config ────────────────────────────
STRICT = True  # True = 缺列立即报错；False = 自动填 NaN

# 设置 Polars 线程数
os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())

# ──────────────────────────── util ────────────────────────────
def require_cols(df: pd.DataFrame, cols: list[str]):
    """严格模式：保证必须列全部存在，否则抛错"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要行情列：{missing}")

# ──────────────────────────── Enhanced Labeling Functions (Pandas) ────────────────────────────
def calculate_order_book_pressure(df_pd: pd.DataFrame) -> pd.DataFrame:
    """计算订单簿压力指标 (Pandas version)"""
    try:
        if '申买量1' in df_pd.columns and '申卖量1' in df_pd.columns:
            df_pd['book_imbalance'] = (df_pd['申买量1'] - df_pd['申卖量1']) / (df_pd['申买量1'] + df_pd['申卖量1'] + 1e-8)
        else:
            df_pd['book_imbalance'] = 0.0
            # console.print("  ⚠️ Missing 申买量1/申卖量1 columns (Pandas), using simplified book_imbalance")
        
        required_cols = ['委托价格', 'bid1', 'ask1', 'is_buy']
        # Ensure 'bid1', 'ask1', 'is_buy' are present from calc_realtime_features
        # Ensure 'spread' is present if needed
        if 'bid1' not in df_pd.columns and '申买价1' in df_pd.columns: df_pd['bid1'] = df_pd['申买价1']
        if 'ask1' not in df_pd.columns and '申卖价1' in df_pd.columns: df_pd['ask1'] = df_pd['申卖价1']
        if 'is_buy' not in df_pd.columns and '方向_委托' in df_pd.columns: df_pd['is_buy'] = (df_pd['方向_委托'] == "买").astype(int)

        missing_cols = [col for col in required_cols if col not in df_pd.columns]
        
        if missing_cols:
            # console.print(f"  ⚠️ Missing columns for price_aggressiveness (Pandas): {missing_cols}")
            df_pd['price_aggressiveness'] = 0.0
        else:
            if 'spread' not in df_pd.columns:
                 if 'bid1' in df_pd.columns and 'ask1' in df_pd.columns:
                    df_pd['spread'] = df_pd['ask1'] - df_pd['bid1']
                 else:
                    df_pd['spread'] = 0.0 # Or handle error

            df_pd['price_aggressiveness'] = np.where(
                df_pd['is_buy'] == 1,
                (df_pd['委托价格'] - df_pd['bid1']) / (df_pd['spread'].replace(0, 1e-8) + 1e-8), # Avoid division by zero in spread
                (df_pd['ask1'] - df_pd['委托价格']) / (df_pd['spread'].replace(0, 1e-8) + 1e-8)
            ).astype(float)
            df_pd['price_aggressiveness'].fillna(0.0, inplace=True)

    except Exception as e:
        # console.print(f"  ⚠️ Error in calculate_order_book_pressure (Pandas): {e}")
        df_pd['book_imbalance'] = df_pd.get('book_imbalance', 0.0)
        df_pd['price_aggressiveness'] = df_pd.get('price_aggressiveness', 0.0)
    
    return df_pd

def detect_layering_pattern(group_df_pd: pd.DataFrame) -> pd.DataFrame:
    """检测分层下单模式（Layering） (Pandas version)"""
    try:
        required_cols = ['委托_datetime', '方向_委托', '委托价格', '委托数量', 'is_buy']
        if not all(col in group_df_pd.columns for col in required_cols):
            # console.print(f"    ⚠️ Missing columns for layering detection (Pandas): {[c for c in required_cols if c not in group_df_pd.columns]}")
            group_df_pd['layering_score'] = 0
            return group_df_pd
        
        # Ensure types
        group_df_pd['委托_datetime'] = pd.to_datetime(group_df_pd['委托_datetime'])
        group_df_pd['委托数量'] = pd.to_numeric(group_df_pd['委托数量'], errors='coerce').fillna(0)
        group_df_pd['委托价格'] = pd.to_numeric(group_df_pd['委托价格'], errors='coerce').fillna(0)


        group_df_pd = group_df_pd.sort_values('委托_datetime')
        layering_signals = []
        
        # Handle case where '委托数量' might be all zeros or very few non-zeros leading to percentile issues
        try:
            median_qty_group = group_df_pd['委托数量'].quantile(0.5) if not group_df_pd['委托数量'].empty else 0
        except Exception:
            median_qty_group = 0

        for i, row in group_df_pd.iterrows():
            time_window = pd.Timedelta('1s')
            start_time = row['委托_datetime'] - time_window
            end_time = row['委托_datetime'] + time_window
            
            nearby_orders = group_df_pd[
                (group_df_pd['委托_datetime'] >= start_time) & 
                (group_df_pd['委托_datetime'] <= end_time) &
                (group_df_pd['方向_委托'] == row['方向_委托'])
            ]
            
            if len(nearby_orders) >= 3:
                prices = nearby_orders['委托价格'].values
                quantities = nearby_orders['委托数量'].values.astype(float) # Ensure float for np operations
                
                if row['is_buy'] == 1:
                    price_ordered = np.all(np.diff(prices) >= 0)
                else:
                    price_ordered = np.all(np.diff(prices) <= 0)
                
                qty_mean = np.mean(quantities) if len(quantities) > 0 else 0
                qty_std = np.std(quantities) if len(quantities) > 0 else 0

                qty_small = qty_mean < median_qty_group
                qty_similar = qty_std / (qty_mean + 1e-8) < 0.5 if qty_mean > 1e-9 else False
                
                layering_score = int(price_ordered and qty_small and qty_similar)
            else:
                layering_score = 0
            layering_signals.append(layering_score)
        
        group_df_pd['layering_score'] = layering_signals
    except Exception as e:
        # console.print(f"    ⚠️ Error in layering detection (Pandas): {e}")
        group_df_pd['layering_score'] = 0
    
    return group_df_pd

def improved_spoofing_rules(df_pd: pd.DataFrame) -> pd.DataFrame:
    """改进的欺诈检测规则 (Pandas version)"""
    labels = {}
    try:
        # Ensure 'bid1', 'ask1' are present for at_bid/at_ask logic
        if 'bid1' not in df_pd.columns and '申买价1' in df_pd.columns: df_pd['bid1'] = df_pd['申买价1']
        if 'ask1' not in df_pd.columns and '申卖价1' in df_pd.columns: df_pd['ask1'] = df_pd['申卖价1']

        if 'at_bid' not in df_pd.columns and all(c in df_pd.columns for c in ['委托价格', 'bid1', '方向_委托']):
            df_pd['at_bid'] = ((df_pd['委托价格'] == df_pd['bid1']) & (df_pd['方向_委托'] == '买')).astype(int)
        elif 'at_bid' not in df_pd.columns:
            df_pd['at_bid'] = 0

        if 'at_ask' not in df_pd.columns and all(c in df_pd.columns for c in ['委托价格', 'ask1', '方向_委托']):
            df_pd['at_ask'] = ((df_pd['委托价格'] == df_pd['ask1']) & (df_pd['方向_委托'] == '卖')).astype(int)
        elif 'at_ask' not in df_pd.columns:
            df_pd['at_ask'] = 0
            
        df_pd['price_aggressiveness'] = df_pd.get('price_aggressiveness', 0.0)
        df_pd['layering_score'] = df_pd.get('layering_score', 0)
            
        if all(col in df_pd.columns for col in ['存活时间_ms', '事件类型', '委托数量', 'ticker', 'at_bid', 'at_ask']):
            conditions_r1 = [
                df_pd['存活时间_ms'] < 100, df_pd['事件类型'] == '撤单',
                (df_pd['at_bid'] == 1) | (df_pd['at_ask'] == 1),
                df_pd['委托数量'] > df_pd.groupby('ticker')['委托数量'].transform('median') * 2
            ]
            labels['quick_cancel_impact'] = np.all(conditions_r1, axis=0).astype(int)
        else: labels['quick_cancel_impact'] = np.zeros(len(df_pd), dtype=int)
        
        if all(col in df_pd.columns for col in ['存活时间_ms', '事件类型', '委托数量', 'ticker', 'price_aggressiveness']):
            conditions_r2 = [
                df_pd['存活时间_ms'] < 500, df_pd['事件类型'] == '撤单',
                np.abs(df_pd['price_aggressiveness']) > 2.0,
                df_pd['委托数量'] > df_pd.groupby('ticker')['委托数量'].transform('quantile', 0.75)
            ]
            labels['price_manipulation'] = np.all(conditions_r2, axis=0).astype(int)
        else: labels['price_manipulation'] = np.zeros(len(df_pd), dtype=int)
        
        if all(col in df_pd.columns for col in ['存活时间_ms', '事件类型', '委托价格', 'bid1', 'ask1', '委托数量', 'ticker']):
            conditions_r3 = [
                df_pd['存活时间_ms'] < 200, df_pd['事件类型'] == '撤单',
                ((df_pd['委托价格'] == df_pd['bid1']) | (df_pd['委托价格'] == df_pd['ask1'])),
                df_pd['委托数量'] > df_pd.groupby('ticker')['委托数量'].transform('quantile', 0.9)
            ]
            labels['fake_liquidity'] = np.all(conditions_r3, axis=0).astype(int)
        else: labels['fake_liquidity'] = np.zeros(len(df_pd), dtype=int)
        
        if all(col in df_pd.columns for col in ['存活时间_ms', '事件类型', 'layering_score']):
            conditions_r4 = [
                df_pd['layering_score'] > 0, df_pd['存活时间_ms'] < 1000, df_pd['事件类型'] == '撤单'
            ]
            labels['layering_cancel'] = np.all(conditions_r4, axis=0).astype(int)
        else: labels['layering_cancel'] = np.zeros(len(df_pd), dtype=int)
        
        if all(col in df_pd.columns for col in ['委托_datetime', '存活时间_ms', '事件类型', '委托数量', 'ticker']):
            df_pd['委托_datetime'] = pd.to_datetime(df_pd['委托_datetime'])
            market_active_hours = (
                (df_pd['委托_datetime'].dt.time >= pd.to_datetime('09:30').time()) &
                (df_pd['委托_datetime'].dt.time <= pd.to_datetime('10:30').time())
            ) | (
                (df_pd['委托_datetime'].dt.time >= pd.to_datetime('14:00').time()) &
                (df_pd['委托_datetime'].dt.time <= pd.to_datetime('15:00').time())
            )
            conditions_r5 = [
                df_pd['存活时间_ms'] < 50, df_pd['事件类型'] == '撤单', market_active_hours,
                df_pd['委托数量'] > df_pd.groupby('ticker')['委托数量'].transform('median')
            ]
            labels['active_hours_spoofing'] = np.all(conditions_r5, axis=0).astype(int)
        else: labels['active_hours_spoofing'] = np.zeros(len(df_pd), dtype=int)
        
        labels_df = pd.DataFrame(labels)
        df_pd['composite_spoofing'] = (
            labels_df['quick_cancel_impact'] | labels_df['price_manipulation'] | 
            labels_df['fake_liquidity'] | labels_df['layering_cancel'] |
            labels_df['active_hours_spoofing']
        ).astype(int)
        df_pd['conservative_spoofing'] = (
            (labels_df['quick_cancel_impact'] + labels_df['price_manipulation'] + 
             labels_df['fake_liquidity'] + labels_df['layering_cancel'] + 
             labels_df['active_hours_spoofing']) >= 2
        ).astype(int)
        
        # Merge individual labels into df_pd before returning
        for col_name in labels_df.columns:
            if col_name not in df_pd.columns:
                 df_pd[col_name] = labels_df[col_name]
            
    except Exception as e:
        # console.print(f"  ⚠️ Error in improved_spoofing_rules (Pandas): {e}")
        default_labels = ['quick_cancel_impact', 'price_manipulation', 'fake_liquidity', 
                          'layering_cancel', 'active_hours_spoofing', 'composite_spoofing', 'conservative_spoofing']
        for label_name in default_labels:
            if label_name not in df_pd.columns:
                df_pd[label_name] = 0
    return df_pd

def analyze_label_quality(df_pd: pd.DataFrame):
    """分析标签质量 (Pandas version)"""
    # console.print("\n📊 Label Quality Analysis (Pandas):")
    label_cols = [col for col in df_pd.columns if 'spoofing' in col or col.startswith('quick_') 
                  or col.startswith('price_') or col.startswith('fake_') 
                  or col.startswith('layering_') or col.startswith('active_')]
    
    for col in label_cols:
        if col in df_pd.columns:
            pos_count = df_pd[col].sum()
            pos_rate = pos_count / len(df_pd) * 100 if len(df_pd) > 0 else 0
            # console.print(f"  {col}: {pos_count:,} ({pos_rate:.4f}%)")
    if len(label_cols) > 1:
        valid_label_cols = [lc for lc in label_cols if lc in df_pd.columns]
        if len(valid_label_cols) > 1 :
            label_corr = df_pd[valid_label_cols].corr()
            # console.print(f"\n📈 Label Correlations (Pandas):")
            # console.print(label_corr.round(3))

# ──────────────────────────── Polars Native Enhanced Labeling Functions ────────────────────────────
def calculate_order_book_pressure_polars(df: pl.LazyFrame, schema: dict) -> pl.LazyFrame:
    """使用 Polars 计算订单簿压力指标"""
    if '申买量1' in schema and '申卖量1' in schema:
        df = df.with_columns(
            ((pl.col('申买量1').cast(pl.Float64) - pl.col('申卖量1').cast(pl.Float64)) / 
             (pl.col('申买量1').cast(pl.Float64) + pl.col('申卖量1').cast(pl.Float64) + 1e-8))
            .fill_null(0.0).alias('book_imbalance')
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias('book_imbalance'))
        # console.print("  ⚠️ Missing 申买量1/申卖量1 columns (Polars), using simplified book_imbalance")

    pa_req_cols = ['委托价格', 'bid1', 'ask1', 'is_buy', 'spread']
    if all(c in schema for c in pa_req_cols):
         df = df.with_columns(
            pl.when(pl.col('is_buy') == 1)
            .then((pl.col('委托价格') - pl.col('bid1')) / (pl.col('spread').replace(0, 1e-8) + 1e-8)) # Avoid division by zero
            .otherwise((pl.col('ask1') - pl.col('委托价格')) / (pl.col('spread').replace(0, 1e-8) + 1e-8))
            .fill_null(0.0).alias('price_aggressiveness')
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias('price_aggressiveness'))
        # missing_pa_cols = [c for c in pa_req_cols if c not in schema]
        # console.print(f"  ⚠️ Missing columns for price_aggressiveness (Polars): {missing_pa_cols}")
    return df

def improved_spoofing_rules_polars(df: pl.LazyFrame, schema: dict) -> pl.LazyFrame:
    """使用 Polars 改进的欺诈检测规则"""
    if '委托价格' in schema and 'bid1' in schema and '方向_委托' in schema:
        df = df.with_columns(((pl.col('委托价格') == pl.col('bid1')) & (pl.col('方向_委托') == '买')).cast(pl.Int8).alias('at_bid'))
    else: df = df.with_columns(pl.lit(0).cast(pl.Int8).alias('at_bid'))

    if '委托价格' in schema and 'ask1' in schema and '方向_委托' in schema:
        df = df.with_columns(((pl.col('委托价格') == pl.col('ask1')) & (pl.col('方向_委托') == '卖')).cast(pl.Int8).alias('at_ask'))
    else: df = df.with_columns(pl.lit(0).cast(pl.Int8).alias('at_ask'))

    if 'price_aggressiveness' not in schema: df = df.with_columns(pl.lit(0.0).alias('price_aggressiveness'))
    if 'layering_score' not in schema: df = df.with_columns(pl.lit(0).cast(pl.Int8).alias('layering_score')) # Must be created before this

    rule1_req = ['存活时间_ms', '事件类型', '委托数量', 'ticker', 'at_bid', 'at_ask']
    if all(c in schema for c in rule1_req) and '委托数量' in schema and 'ticker' in schema:
        rule1_cond = ((pl.col('存活时间_ms') < 100) & (pl.col('事件类型') == '撤单') &
                      ((pl.col('at_bid') == 1) | (pl.col('at_ask') == 1)) &
                      (pl.col('委托数量') > pl.col('委托数量').median().over('ticker') * 2))
        df = df.with_columns(rule1_cond.cast(pl.Int8).alias('quick_cancel_impact'))
    else: df = df.with_columns(pl.lit(0).cast(pl.Int8).alias('quick_cancel_impact'))

    rule2_req = ['存活时间_ms', '事件类型', '委托数量', 'ticker', 'price_aggressiveness']
    if all(c in schema for c in rule2_req) and '委托数量' in schema and 'ticker' in schema:
        rule2_cond = ((pl.col('存活时间_ms') < 500) & (pl.col('事件类型') == '撤单') &
                      (pl.col('price_aggressiveness').abs() > 2.0) &
                      (pl.col('委托数量') > pl.col('委托数量').quantile(0.75).over('ticker')))
        df = df.with_columns(rule2_cond.cast(pl.Int8).alias('price_manipulation'))
    else: df = df.with_columns(pl.lit(0).cast(pl.Int8).alias('price_manipulation'))

    rule3_req = ['存活时间_ms', '事件类型', '委托价格', 'bid1', 'ask1', '委托数量', 'ticker']
    if all(c in schema for c in rule3_req) and '委托数量' in schema and 'ticker' in schema:
        rule3_cond = ((pl.col('存活时间_ms') < 200) & (pl.col('事件类型') == '撤单') &
                      ((pl.col('委托价格') == pl.col('bid1')) | (pl.col('委托价格') == pl.col('ask1'))) &
                      (pl.col('委托数量') > pl.col('委托数量').quantile(0.90).over('ticker')))
        df = df.with_columns(rule3_cond.cast(pl.Int8).alias('fake_liquidity'))
    else: df = df.with_columns(pl.lit(0).cast(pl.Int8).alias('fake_liquidity'))

    rule4_req = ['存活时间_ms', '事件类型', 'layering_score']
    if all(c in schema for c in rule4_req):
        rule4_cond = ((pl.col('layering_score') > 0) & (pl.col('存活时间_ms') < 1000) & (pl.col('事件类型') == '撤单'))
        df = df.with_columns(rule4_cond.cast(pl.Int8).alias('layering_cancel'))
    else: df = df.with_columns(pl.lit(0).cast(pl.Int8).alias('layering_cancel'))

    rule5_req = ['委托_datetime', '存活时间_ms', '事件类型', '委托数量', 'ticker']
    if all(c in schema for c in rule5_req) and '委托数量' in schema and 'ticker' in schema:
        market_active_hours = (((pl.col('委托_datetime').dt.time() >= pl.time(9, 30, 0)) & (pl.col('委托_datetime').dt.time() <= pl.time(10, 30, 0))) |
                               ((pl.col('委托_datetime').dt.time() >= pl.time(14, 0, 0)) & (pl.col('委托_datetime').dt.time() <= pl.time(15, 0, 0))))
        rule5_cond = ((pl.col('存活时间_ms') < 50) & (pl.col('事件类型') == '撤单') & market_active_hours &
                      (pl.col('委托数量') > pl.col('委托数量').median().over('ticker')))
        df = df.with_columns(rule5_cond.cast(pl.Int8).alias('active_hours_spoofing'))
    else: df = df.with_columns(pl.lit(0).cast(pl.Int8).alias('active_hours_spoofing'))

    df = df.with_columns(
        (pl.col('quick_cancel_impact').cast(pl.Boolean) | pl.col('price_manipulation').cast(pl.Boolean) |
         pl.col('fake_liquidity').cast(pl.Boolean) | pl.col('layering_cancel').cast(pl.Boolean) |
         pl.col('active_hours_spoofing').cast(pl.Boolean)
        ).cast(pl.Int8).alias('composite_spoofing')
    )
    df = df.with_columns(
        ((pl.col('quick_cancel_impact') + pl.col('price_manipulation') + pl.col('fake_liquidity') +
          pl.col('layering_cancel') + pl.col('active_hours_spoofing')) >= 2
        ).cast(pl.Int8).alias('conservative_spoofing')
    )
    df = df.with_columns(pl.col('composite_spoofing').alias('y_label'))
    return df

# ──────────────────────────── Original Functions ────────────────────────────
def calc_realtime_features(df_pd: pd.DataFrame) -> pd.DataFrame: # Renamed to df_pd
    """实时可观测特征列 (Pandas version)"""
    needed = ["申买价1", "申卖价1", "前收盘"]
    if STRICT:
        # Ensure cols exist or create them with NaN if not STRICT (but STRICT is True)
        for col in needed:
            if col not in df_pd.columns: df_pd[col] = np.nan # Or raise error based on STRICT
        # require_cols(df_pd, needed) # This would raise error

    df_pd["bid1"] = df_pd.get("申买价1")
    df_pd["ask1"] = df_pd.get("申卖价1")
    df_pd["prev_close"] = df_pd.get("前收盘")

    df_pd["is_buy"] = (df_pd["方向_委托"] == "买").astype(int)
    df_pd["mid_price"] = (df_pd["bid1"] + df_pd["ask1"]) / 2
    df_pd["spread"] = df_pd["ask1"] - df_pd["bid1"]
    df_pd["delta_mid"] = df_pd["委托价格"] - df_pd["mid_price"]
    # Ensure spread is not zero for division
    df_pd["pct_spread"] = (df_pd["delta_mid"].abs() / df_pd["spread"].replace(0, 1e-8)).replace([np.inf,-np.inf],0).fillna(0)
    df_pd["log_qty"] = np.log1p(df_pd["委托数量"])

    df_pd.sort_values("委托_datetime", inplace=True)
    df_pd["orders_100ms"] = (df_pd.rolling("100ms", on="委托_datetime")["委托_datetime"].count().shift(1).fillna(0))
    df_pd["is_cancel"] = (df_pd["事件类型"] == "撤单").astype(int)
    df_pd["cancels_5s"] = (df_pd.rolling("5s", on="委托_datetime")["is_cancel"].sum().shift(1).fillna(0))

    # Ensure '委托_datetime' is datetime
    df_pd['委托_datetime'] = pd.to_datetime(df_pd['委托_datetime'])
    sec = (df_pd["委托_datetime"] - df_pd["委托_datetime"].dt.normalize() - pd.Timedelta("09:30:00")).dt.total_seconds()
    df_pd["time_sin"] = np.sin(2*np.pi*sec/(4.5*3600))
    df_pd["time_cos"] = np.cos(2*np.pi*sec/(4.5*3600))

    df_pd["in_auction"] = (((df_pd["委托_datetime"].dt.time >= pd.to_datetime("09:15").time()) & (df_pd["委托_datetime"].dt.time < pd.to_datetime("09:30").time())) |
                         ((df_pd["委托_datetime"].dt.time >= pd.to_datetime("14:57").time()) & (df_pd["委托_datetime"].dt.time <= pd.to_datetime("15:00").time()))).astype(int)
    df_pd["price_dev_prevclose"] = ((df_pd["委托价格"] - df_pd["prev_close"]) / df_pd["prev_close"].replace(0,1e-8)).fillna(0)
    return df_pd

def apply_label_rules(df_pd: pd.DataFrame, r1_ms: int, r2_ms: int, r2_mult: float): # Renamed to df_pd
    """修正版标签规则 (Pandas version)"""
    df_pd["flag_R1"] = (df_pd["存活时间_ms"] < r1_ms) & (df_pd["事件类型"] == "撤单")
    safe_spread = df_pd["spread"].fillna(np.inf).replace(0, np.inf) # Avoid division by zero or NaN issues
    df_pd["flag_R2"] = ((df_pd["存活时间_ms"] < r2_ms) & (df_pd["delta_mid"].abs() >= r2_mult * safe_spread))
    df_pd["y_label"] = (df_pd["flag_R1"] & df_pd["flag_R2"]).astype(int)
    return df_pd

def calc_realtime_features_polars(df: pl.LazyFrame) -> pl.LazyFrame:
    """使用 Polars Lazy API 计算实时可观测特征"""
    df = df.with_columns([
        pl.col("委托_datetime").cast(pl.Datetime("ns")).alias("ts_ns"),
        pl.col("事件_datetime").cast(pl.Datetime("ns")).alias("event_ts_ns"),
    ])
    
    df = df.with_columns([
        pl.col("申买价1").alias("bid1").cast(pl.Float64),
        pl.col("申卖价1").alias("ask1").cast(pl.Float64),
        pl.col("前收盘").alias("prev_close").cast(pl.Float64),
        (pl.col("方向_委托") == "买").cast(pl.Int8).alias("is_buy"),
        (pl.col("事件类型") == "撤单").cast(pl.Int8).alias("is_cancel"),
        pl.col("委托数量").log1p().alias("log_qty"), # Assumes 委托数量 is numeric
        pl.col("委托价格").cast(pl.Float64), # Ensure 委托价格 is float
    ])
    
    df = df.with_columns([
        ((pl.col("bid1") + pl.col("ask1")) / 2).alias("mid_price"),
        (pl.col("ask1") - pl.col("bid1")).alias("spread"),
    ])
    
    df = df.with_columns([
        (pl.col("委托价格") - pl.col("mid_price")).alias("delta_mid"),
        ((pl.col("委托价格") - pl.col("prev_close")) / pl.col("prev_close").replace(0, 1e-8)).fill_null(0).alias("price_dev_prevclose"),
    ])
    
    df = df.with_columns([
        (pl.col("delta_mid").abs() / pl.col("spread").replace(0,1e-8)).fill_null(0).replace(float("inf"), 0).replace(float("-inf"),0).alias("pct_spread"),
    ])
    
    df = df.with_columns([
        ((pl.col("委托_datetime").dt.hour() - 9) * 3600 + 
         (pl.col("委托_datetime").dt.minute() - 30) * 60 + 
         pl.col("委托_datetime").dt.second()).alias("seconds_since_market_open"),
        (pl.col("委托_datetime").dt.time().is_between(pl.time(9,15), pl.time(9,30), "left") |
         pl.col("委托_datetime").dt.time().is_between(pl.time(14,57), pl.time(15,0), "both")
        ).cast(pl.Int8).alias("in_auction"),
    ])
    
    df = df.with_columns([
        (2 * np.pi * pl.col("seconds_since_market_open") / (4.5 * 3600)).sin().alias("time_sin"), # np.pi is fine
        (2 * np.pi * pl.col("seconds_since_market_open") / (4.5 * 3600)).cos().alias("time_cos"),
    ])
    
    # Rolling window features: these are complex with groupby_dynamic and might need collection or simpler alternatives if errors persist.
    # For simplicity and robustness, we might do these on collected data or accept simpler versions.
    # The original code had a try-except block suggesting potential issues.
    # Let's ensure 'ts_ns' is sorted if doing rolling operations.
    # df = df.sort("ts_ns") # scan_csv doesn't guarantee order for groupby_dynamic unless source is sorted.
    # However, subsequent operations might reorder, so this sort needs to be at the right place.
    # Given the previous issues, let's make these robust with defaults if they fail.
    try:
        # Attempting a more robust way to do rolling counts in lazy Polars
        # This still might be tricky without eager execution for precise windowing as in pandas
        df = df.with_columns([
            pl.col("ts_ns").set_sorted().alias("ts_ns_sorted") # Indicate it's sorted for rolling
        ])

        orders_100ms_expr = pl.count().rolling(index_column="ts_ns_sorted", period="100ms").shift(1).fill_null(0)
        cancels_5s_expr = pl.col("is_cancel").sum().rolling(index_column="ts_ns_sorted", period="5s").shift(1).fill_null(0)
        
        df = df.with_columns([
            orders_100ms_expr.alias("orders_100ms"),
            cancels_5s_expr.alias("cancels_5s"),
        ]).drop("ts_ns_sorted")

    except Exception as e:
        # console.print(f"[yellow]Warning: Polars rolling window calculation failed ({e}), using default values[/yellow]")
        df = df.with_columns([
            pl.lit(0).cast(pl.UInt32).alias("orders_100ms"), # Match count type if possible
            pl.lit(0).cast(pl.Int8).alias("cancels_5s"),   # Match sum type
        ])
    return df

def process_one_day_polars(evt_csv: Path, feat_dir: Path, lbl_dir: Path,
                          watch: set, r1_ms: int, r2_ms: int, r2_mult: float,
                          enhanced_labels: bool = False):
    """使用 Polars 处理单日数据"""
    date_str = evt_csv.parent.name
    
    progress_bar_config = [
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(),
    ]

    with Progress(*progress_bar_config, console=console) as progress:
        load_task = progress.add_task(f"[cyan]{date_str}[/cyan] Loading CSV...", total=100)
        try:
            df_lazy = pl.scan_csv(evt_csv, try_parse_dates=True, infer_schema_length=20000,
                                 null_values=["", "NULL", "null", "U"], ignore_errors=True)
            initial_schema = df_lazy.schema 
            progress.update(load_task, advance=100)
        except Exception as e:
            # console.print(f"[yellow]Warning: Polars CSV parsing failed for {date_str}, falling back to pandas ({e})[/yellow]")
            return process_one_day_pandas_fallback(evt_csv, feat_dir, lbl_dir, watch, r1_ms, r2_ms, r2_mult, enhanced_labels)
            
        if watch: df_lazy = df_lazy.filter(pl.col("ticker").is_in(list(watch)))
        df_lazy = df_lazy.with_columns(pl.lit(date_str).alias("自然日"))
        
        feat_task = progress.add_task(f"[green]{date_str}[/green] Computing features...", total=100)
        df_lazy = calc_realtime_features_polars(df_lazy)
        current_schema = df_lazy.schema # Schema after realtime features
        progress.advance(feat_task, 25)

        if enhanced_labels:
            # console.print(f"[blue]Using Polars-native (or mixed) enhanced labeling for {date_str}[/blue]")
            df_lazy = calculate_order_book_pressure_polars(df_lazy, current_schema)
            current_schema = df_lazy.schema 
            progress.advance(feat_task, 25)

            # Layering: Collect, apply pandas detect_layering_pattern, join back
            # console.print(f"    Calculating layering score for {date_str} (mixed Polars/Pandas approach)...", end="")
            temp_df_for_layering = df_lazy.collect(streaming=True) # Use streaming if possible
            
            if not temp_df_for_layering.is_empty() and all(c in temp_df_for_layering.columns for c in ['ticker', '自然日', '委托_datetime', '方向_委托', '委托价格', '委托数量']):
                pdf_for_layering = temp_df_for_layering.to_pandas()
                pdf_for_layering['is_buy'] = (pdf_for_layering['方向_委托'] == "买").astype(int) # Ensure is_buy for pandas function
                
                pdf_layering_groups = []
                for (ticker_val, day_val), group_pdf in pdf_for_layering.groupby(['ticker', '自然日']):
                    group_with_layering = detect_layering_pattern(group_pdf.copy()) # Uses the pandas function
                    pdf_layering_groups.append(group_with_layering)
                
                if pdf_layering_groups:
                    pdf_with_layering = pd.concat(pdf_layering_groups, ignore_index=True)
                    # Create a unique ID in both before converting pdf_with_layering to polars and joining
                    # For simplicity, assuming row order is preserved or select specific columns
                    # Let's ensure 'layering_score' is the only new column or select carefully
                    polars_layering_scores = pl.from_pandas(pdf_with_layering[['layering_score']]) # Get only the score
                    
                    # This join needs to be careful about row alignment. If row count matches, can hstack.
                    if len(temp_df_for_layering) == len(polars_layering_scores):
                         df_lazy = temp_df_for_layering.with_columns(polars_layering_scores['layering_score']).lazy()
                    else: # Fallback if row counts mismatch, add default score
                         # console.print(f"[yellow] Layering score join mismatch for {date_str}, defaulting layering_score.[/yellow]")
                         df_lazy = temp_df_for_layering.with_columns(pl.lit(0).cast(pl.Int8).alias('layering_score')).lazy()
                else:
                    df_lazy = temp_df_for_layering.with_columns(pl.lit(0).cast(pl.Int8).alias('layering_score')).lazy()
            else: # If df was empty or missing columns for layering
                df_lazy = df_lazy.with_columns(pl.lit(0).cast(pl.Int8).alias('layering_score'))
            # console.print(" done.")
            current_schema = df_lazy.schema
            progress.advance(feat_task, 25)
            
            df_lazy = improved_spoofing_rules_polars(df_lazy, current_schema)
            # y_label is 'composite_spoofing' from improved_spoofing_rules_polars
            # Analyze label quality (Pandas for now)
            # df_collected_final_labels = df_lazy.collect(streaming=True)
            # analyze_label_quality(df_collected_final_labels.to_pandas())
        else:
            # Original simple Polars labeling
            df_lazy = df_lazy.with_columns([
                ((pl.col("存活时间_ms") < r1_ms) & (pl.col("事件类型") == "撤单")).cast(pl.Int8).alias("flag_R1"),
                ((pl.col("存活时间_ms") < r2_ms) & (pl.col("delta_mid").abs() >= r2_mult * pl.col("spread").fill_null(float("inf")).replace(0, float("inf")))).cast(pl.Int8).alias("flag_R2"),
            ])
            df_lazy = df_lazy.with_columns((pl.col("flag_R1") & pl.col("flag_R2")).cast(pl.Int8).alias("y_label"))
        progress.update(feat_task, advance=100, total=100) # Mark as complete
        
        agg_task = progress.add_task(f"[yellow]{date_str}[/yellow] Aggregating orders...", total=100)
        df_collected = df_lazy.collect(streaming=True)
        
        if df_collected.is_empty(): return 0,0

        agg_expressions = [
            pl.first(col).alias(col) for col in [
                "bid1", "ask1", "prev_close", "mid_price", "spread", "delta_mid", "pct_spread",
                "orders_100ms", "cancels_5s", "log_qty", "time_sin", "time_cos", "in_auction",
                "price_dev_prevclose", "is_buy", "is_cancel"
            ] if col in df_collected.columns
        ] + [
            pl.count().alias("total_events"),
            pl.col("成交数量").filter(pl.col("事件类型") == "成交").sum().fill_null(0).alias("total_traded_qty"),
            pl.col("事件类型").filter(pl.col("事件类型") == "成交").count().alias("num_trades"),
            pl.col("事件类型").filter(pl.col("事件类型") == "撤单").count().alias("num_cancels"),
            (pl.col("event_ts_ns").max() - pl.col("ts_ns").first()).dt.total_milliseconds().alias("final_survival_time_ms"),
            ((pl.col("成交数量").filter(pl.col("事件类型") == "成交").sum().fill_null(0) >= 
              pl.col("委托数量").first() * 0.99)).cast(pl.Int8).alias("is_fully_filled")
        ]

        if enhanced_labels:
            enhanced_label_cols = ['quick_cancel_impact', 'price_manipulation', 'fake_liquidity', 
                                   'layering_cancel', 'active_hours_spoofing', 'composite_spoofing', 
                                   'conservative_spoofing', 'y_label'] # y_label is composite_spoofing
            for lc in enhanced_label_cols:
                if lc in df_collected.columns: agg_expressions.append(pl.col(lc).max().alias(lc))
                else: agg_expressions.append(pl.lit(0).cast(pl.Int8).alias(lc)) # Ensure col exists

            enhanced_feature_cols = ['book_imbalance', 'price_aggressiveness', 'layering_score']
            for efc in enhanced_feature_cols:
                if efc in df_collected.columns: agg_expressions.append(pl.first(efc).alias(efc))
                else: agg_expressions.append(pl.lit(0.0 if efc != 'layering_score' else 0).alias(efc))
        else: # Simple label
            if "y_label" in df_collected.columns: agg_expressions.append(pl.col("y_label").max().alias("y_label"))
            else: agg_expressions.append(pl.lit(0).cast(pl.Int8).alias("y_label"))
            
        df_orders = df_collected.group_by(["自然日", "ticker", "交易所委托号"]).agg(agg_expressions)
        progress.update(agg_task, advance=100)
        
        save_task = progress.add_task(f"[magenta]{date_str}[/magenta] Saving files...", total=100)
        if df_orders.is_empty(): return 0, 0
        
        feat_cols_base = ["自然日", "交易所委托号", "ticker", "bid1", "ask1", "prev_close", "mid_price", "spread", 
                          "delta_mid", "pct_spread", "orders_100ms", "cancels_5s", "log_qty", "time_sin", 
                          "time_cos", "in_auction", "price_dev_prevclose", "is_buy", "is_cancel", 
                          "total_events", "total_traded_qty", "num_trades", "num_cancels", 
                          "final_survival_time_ms", "is_fully_filled"]
        
        final_feat_cols = [c for c in feat_cols_base if c in df_orders.columns]
        
        label_cols_base = ["自然日", "交易所委托号", "ticker", "y_label"]
        final_label_cols = [c for c in label_cols_base if c in df_orders.columns]

        if enhanced_labels:
            enhanced_feat_to_save = ['book_imbalance', 'price_aggressiveness', 'layering_score']
            final_feat_cols.extend([c for c in enhanced_feat_to_save if c in df_orders.columns])
            
            enhanced_labels_to_save = ['quick_cancel_impact', 'price_manipulation', 'fake_liquidity', 
                                       'layering_cancel', 'active_hours_spoofing', 'composite_spoofing', 
                                       'conservative_spoofing']
            final_label_cols.extend([c for c in enhanced_labels_to_save if c in df_orders.columns])

        feat_dir.mkdir(exist_ok=True); lbl_dir.mkdir(exist_ok=True)
        df_orders[final_feat_cols].write_parquet(feat_dir / f"X_{date_str}.parquet", compression="snappy")
        progress.advance(save_task, 50)
        df_orders[final_label_cols].write_parquet(lbl_dir / f"labels_{date_str}.parquet", compression="snappy")
        progress.advance(save_task, 50)
        
        return len(df_orders), int(df_orders["y_label"].sum() if "y_label" in df_orders.columns else 0)

def process_one_day_pandas_fallback(evt_csv: Path, feat_dir: Path, lbl_dir: Path,
                                   watch: set, r1_ms: int, r2_ms: int, r2_mult: float,
                                   enhanced_labels: bool = False):
    """Pandas 回退处理函数"""
    date_str = evt_csv.parent.name
    progress_bar_config = [
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(),
    ]
    with Progress(*progress_bar_config, console=console, transient=True) as progress: # Transient for fallback
        load_task = progress.add_task(f"[yellow]{date_str}[/yellow] (Pandas Fallback) Loading...", total=100)
        df_pd = pd.read_csv(evt_csv, parse_dates=["委托_datetime","事件_datetime"], low_memory=False)
        progress.update(load_task, advance=100)
        
        if watch: df_pd = df_pd[df_pd["ticker"].isin(watch)]
        if df_pd.empty: return 0, 0
        df_pd["自然日"] = date_str
        
        feat_task = progress.add_task(f"[yellow]{date_str}[/yellow] (Pandas Fallback) Features...", total=100)
        df_pd = calc_realtime_features(df_pd) # Pandas version
        
        if enhanced_labels:
            # console.print(f"  Using enhanced labeling (Pandas fallback) for {date_str}")
            df_pd = calculate_order_book_pressure(df_pd) # Pandas version
            
            enhanced_groups = []
            if not df_pd.empty and all(c in df_pd.columns for c in ['ticker', '自然日', '委托_datetime', '方向_委托', '委托价格', '委托数量']):
                df_pd['is_buy'] = (df_pd['方向_委托'] == "买").astype(int) # Ensure is_buy for pandas function
                for (ticker, date_val), group in df_pd.groupby(['ticker', '自然日']): # Use date_val to avoid conflict
                    group_enhanced = detect_layering_pattern(group.copy()) # Pandas version
                    enhanced_groups.append(group_enhanced)
            
            if enhanced_groups: df_pd = pd.concat(enhanced_groups, ignore_index=True)
            else: df_pd['layering_score'] = 0 # Default if no groups or layering failed
            
            df_pd = improved_spoofing_rules(df_pd) # Pandas version
            df_pd["y_label"] = df_pd.get("composite_spoofing", 0) # Default if col missing
            # analyze_label_quality(df_pd) # Pandas version
        else:
            df_pd = apply_label_rules(df_pd, r1_ms, r2_ms, r2_mult) # Pandas version
        progress.update(feat_task, advance=100)
            
        agg_task = progress.add_task(f"[yellow]{date_str}[/yellow] (Pandas Fallback) Aggregating...", total=100)
        order_groups = df_pd.groupby(["自然日", "ticker", "交易所委托号"])
        order_features = []
        for (date_val, ticker_val, order_id), group in order_groups:
            first_row = group.iloc[0].copy()
            first_row["y_label"] = group["y_label"].max() if "y_label" in group else 0
            first_row["total_events"] = len(group)
            first_row["total_traded_qty"] = group[group["事件类型"] == "成交"]["成交数量"].astype(float).sum()
            first_row["num_trades"] = (group["事件类型"] == "成交").sum()
            first_row["num_cancels"] = (group["事件类型"] == "撤单").sum()
            first_row["final_survival_time_ms"] = (group["事件_datetime"].max() - group["委托_datetime"].iloc[0]).total_seconds() * 1000
            total_order_qty = float(first_row.get("委托数量", 0))
            first_row["is_fully_filled"] = int(first_row["total_traded_qty"] >= total_order_qty * 0.99)
            
            if enhanced_labels:
                for label_col in ['quick_cancel_impact', 'price_manipulation', 'fake_liquidity', 
                                 'layering_cancel', 'active_hours_spoofing', 'composite_spoofing', 
                                 'conservative_spoofing']:
                    if label_col in group.columns:
                        first_row[label_col] = group[label_col].max() if 'spoofing' in label_col or 'cancel' in label_col or 'score' in label_col else group[label_col].iloc[0]
                    else: # Ensure column exists if it was meant to be there
                        first_row[label_col] = 0.0 if 'book' in label_col or 'aggres' in label_col else 0
            order_features.append(first_row)
        
        df_orders = pd.DataFrame(order_features)
        progress.update(agg_task, advance=100)
        
        save_task = progress.add_task(f"[yellow]{date_str}[/yellow] (Pandas Fallback) Saving...", total=100)
        if df_orders.empty: return 0, 0
        
        feat_cols_base = ["自然日", "交易所委托号","ticker","bid1","ask1","prev_close","mid_price","spread","delta_mid","pct_spread","orders_100ms","cancels_5s","log_qty","time_sin","time_cos","in_auction","price_dev_prevclose","is_buy","is_cancel","total_events", "total_traded_qty", "num_trades", "num_cancels","final_survival_time_ms", "is_fully_filled"]
        final_feat_cols = [c for c in feat_cols_base if c in df_orders.columns]
        label_cols_base = ["自然日","交易所委托号","ticker","y_label"]
        final_label_cols = [c for c in label_cols_base if c in df_orders.columns]

        if enhanced_labels:
            enhanced_feat_to_save = ['book_imbalance', 'price_aggressiveness', 'layering_score']
            final_feat_cols.extend([c for c in enhanced_feat_to_save if c in df_orders.columns])
            enhanced_labels_to_save = ['quick_cancel_impact', 'price_manipulation', 'fake_liquidity', 'layering_cancel', 'active_hours_spoofing', 'composite_spoofing', 'conservative_spoofing']
            final_label_cols.extend([c for c in enhanced_labels_to_save if c in df_orders.columns])
        
        # Ensure all selected columns actually exist in df_orders before saving
        final_feat_cols = [c for c in final_feat_cols if c in df_orders.columns]
        final_label_cols = [c for c in final_label_cols if c in df_orders.columns]


        feat_dir.mkdir(exist_ok=True); lbl_dir.mkdir(exist_ok=True)
        df_orders[final_feat_cols].to_parquet(feat_dir / f"X_{date_str}.parquet", index=False)
        progress.advance(save_task, 50)
        df_orders[final_label_cols].to_parquet(lbl_dir / f"labels_{date_str}.parquet", index=False)
        progress.advance(save_task, 50)
    
    return len(df_orders), int(df_orders["y_label"].sum() if "y_label" in df_orders.columns else 0)


def process_one_day(evt_csv: Path, feat_dir: Path, lbl_dir: Path,
                    watch: set, r1_ms: int, r2_ms: int, r2_mult: float,
                    backend: str = "pandas", enhanced_labels: bool = False):
    """根据后端选择处理函数"""
    if backend == "polars":
        return process_one_day_polars(evt_csv, feat_dir, lbl_dir, watch, r1_ms, r2_ms, r2_mult, enhanced_labels)
    else: # Pandas backend
        date_str = evt_csv.parent.name
        # console.print(f"[blue]Using Pandas backend for {date_str} (enhanced_labels={enhanced_labels})[/blue]")
        return process_one_day_pandas_fallback(evt_csv, feat_dir, lbl_dir, watch, r1_ms, r2_ms, r2_mult, enhanced_labels)


# ───────────────────────────── main ────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="event_stream 根目录")
    parser.add_argument("--tickers", nargs="*", default=None, help="股票白名单")
    parser.add_argument("--r1_ms", type=int, default=50)
    parser.add_argument("--r2_ms", type=int, default=1000)
    parser.add_argument("--r2_mult", type=float, default=4.0)
    parser.add_argument("--backend", choices=["pandas", "polars"], default="polars")
    parser.add_argument("--max_workers", type=int, default=None, help="进程池最大工作进程数")
    parser.add_argument("--enhanced_labels", action="store_true", help="使用增强标签")
    args = parser.parse_args()

    evt_root = Path(args.root)
    parent   = evt_root.parent
    feat_dir = parent / "features_select"
    lbl_dir  = parent / "labels_select"
    watch    = set(args.tickers) if args.tickers else set()

    date_dirs = [p for p in evt_root.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)]
    if not date_dirs: sys.exit("[ERR] no date folders under event_stream")

    console.print(f"\n[bold green]🚀 Starting ETL Pipeline[/bold green]")
    console.print(f"[dim]Root: {evt_root}[/dim]")
    console.print(f"[dim]Backend: {args.backend}[/dim]")
    console.print(f"[dim]Using Enhanced Labels: {args.enhanced_labels}[/dim]")
    console.print(f"[dim]Tickers: {list(watch) if watch else 'All'}[/dim]")
    console.print(f"[dim]Max Workers: {args.max_workers if args.max_workers else 'Default (Polars backend sequential, Pandas backend sequential)'}[/dim]")
    console.print(f"[dim]Found {len(date_dirs)} date directories[/dim]\n")
    
    total_rows, total_positives, successful_days = 0, 0, 0
    
    # For Polars, ProcessPoolExecutor is mainly for IO-bound tasks or true sub-process parallelism.
    # Polars itself parallelizes computations using its own thread pool (POLARS_MAX_THREADS).
    # If max_workers is for date-level parallelism, then it's fine.
    
    actual_max_workers = args.max_workers
    if args.backend == 'polars' and args.max_workers is None:
        # If polars and no max_workers, let polars handle its internal parallelism, run days sequentially here.
        actual_max_workers = 1 
    elif args.backend == 'pandas' and args.max_workers is None:
        actual_max_workers = 1 # Pandas runs sequentially per day by default

    if actual_max_workers and actual_max_workers > 1 :
        with ProcessPoolExecutor(max_workers=actual_max_workers) as executor:
            futures = {
                executor.submit(process_one_day, d / "委托事件流.csv", feat_dir, lbl_dir, watch, 
                                args.r1_ms, args.r2_ms, args.r2_mult, args.backend, args.enhanced_labels): d
                for d in sorted(date_dirs) if (d / "委托事件流.csv").exists()
            }
            
            # Print initial missing files if any
            for d_path in sorted(date_dirs):
                 if not (d_path / "委托事件流.csv").exists():
                      rprint(f"[yellow]× {d_path.name}: missing 委托事件流.csv (skipped)[/yellow]")

            for future in futures: # Iterate over keys (futures)
                d = futures[future] # Get corresponding date_dir
                try:
                    rows, pos = future.result()
                    if rows > 0 : # Check if any orders were processed
                        rprint(f"[green]✓ {d.name}: orders={rows:,}  positive={pos}[/green]")
                        total_rows += rows; total_positives += pos; successful_days += 1
                    elif rows == 0 and pos == 0 : # No data for watchlist or empty after processing
                        rprint(f"[yellow]~ {d.name}: no watch-list data or empty result[/yellow]")
                    # else: rows could be None if error in process_one_day and not handled to return tuple
                except Exception as e:
                    rprint(f"[red]× {d.name}: ERROR in future - {type(e).__name__}: {str(e).replace('[', '\\[').replace(']', '\\]')}[/red]")
    else: # Sequential processing
        for d in sorted(date_dirs):
            evt_csv = d / "委托事件流.csv"
            if not evt_csv.exists():
                rprint(f"[yellow]× {d.name}: missing 委托事件流.csv (skipped)[/yellow]")
                continue
            try:
                rows, pos = process_one_day(evt_csv, feat_dir, lbl_dir, watch, 
                                          args.r1_ms, args.r2_ms, args.r2_mult, 
                                          args.backend, args.enhanced_labels)
                if rows > 0:
                    rprint(f"[green]✓ {d.name}: orders={rows:,}  positive={pos}[/green]")
                    total_rows += rows; total_positives += pos; successful_days += 1
                elif rows == 0 and pos == 0:
                     rprint(f"[yellow]~ {d.name}: no watch-list data or empty result[/yellow]")
            except Exception as e:
                rprint(f"[red]× {d.name}: ERROR - {type(e).__name__}: {str(e).replace('[', '\\[').replace(']', '\\]')}[/red]")
                continue
    
    console.print(f"\n[bold cyan]📊 ETL Pipeline Summary[/bold cyan]")
    console.print(f"  Backend: {args.backend}")
    console.print(f"  Using Enhanced Labels: {args.enhanced_labels}")
    console.print(f"  Processed days: {successful_days}/{len(date_dirs)}")
    console.print(f"  Total orders: {total_rows:,}")
    console.print(f"  Positive labels: {total_positives:,}")
    if total_rows > 0: console.print(f"  Positive rate: {total_positives / total_rows * 100:.4f}%")
    console.print(f"[bold green]✅ ETL Pipeline completed![/bold green]\n")

if __name__ == "__main__":
    main()

"""
# 使用 Polars 后端（推荐）
python scripts/data_process/run_etl_from_event.py \
    --root "/obs/users/fenglang/general/Spoofing Detect/data/event_stream" \
    --tickers 000989.SZ 300233.SZ \
    --r1_ms 100 --r2_ms 1000 --r2_mult 1.5 \
    --backend polars \
    --max_workers 10 \
    --enhanced_labels # 使用增强标签

# 使用 Pandas 后端（兼容）
python scripts/data_process/run_etl_from_event.py \
    --root "/obs/users/fenglang/general/Spoofing Detect/data/event_stream" \
    --tickers 000989.SZ 300233.SZ \
    --r1_ms 50 --r2_ms 1000 --r2_mult 4.0 \
    --backend pandas \
    --enhanced_labels # 使用增强标签
"""