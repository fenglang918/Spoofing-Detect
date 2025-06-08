#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_engineering.py
────────────────────────────────────────
特征工程模块 - 包含所有特征计算相关的函数
"""

import pandas as pd
import numpy as np
import polars as pl
from rich.console import Console
from typing import List, Dict

try:
    from .utils import STRICT
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from utils import STRICT

console = Console()

# ──────────────────────────── Pandas Feature Engineering ────────────────────────────
def calc_realtime_features(df_pd: pd.DataFrame) -> pd.DataFrame:
    """实时可观测特征列 (Pandas version) - 严格防泄露版：仅使用委托时刻及之前的信息"""
    needed = ["申买价1", "申卖价1", "前收盘", "申买量1", "申卖量1"]
    if STRICT:
        # Ensure cols exist or create them with NaN if not STRICT (but STRICT is True)
        for col in needed:
            if col not in df_pd.columns: 
                df_pd[col] = np.nan

    # 基础盘口特征
    df_pd["bid1"] = df_pd.get("申买价1")
    df_pd["ask1"] = df_pd.get("申卖价1")
    df_pd["prev_close"] = df_pd.get("前收盘")
    df_pd["bid_vol1"] = df_pd.get("申买量1", 0)
    df_pd["ask_vol1"] = df_pd.get("申卖量1", 0)

    df_pd["is_buy"] = (df_pd["方向_委托"] == "买").astype(int)
    df_pd["mid_price"] = (df_pd["bid1"] + df_pd["ask1"]) / 2
    df_pd["spread"] = df_pd["ask1"] - df_pd["bid1"]
    
    # 修正 delta_mid：使用当前时刻的价格减去当前时刻的mid
    df_pd["delta_mid"] = df_pd["委托价格"] - df_pd["mid_price"]
    
    # 修正 pct_spread：防止溢出，使用更稳定的计算
    safe_mid = df_pd["mid_price"].replace(0, np.nan)
    df_pd["pct_spread"] = (df_pd["delta_mid"].abs() / safe_mid * 100).fillna(0)
    # 截断异常值
    df_pd["pct_spread"] = df_pd["pct_spread"].clip(-500, 500)
    
    # 对数变换：处理尺度过大的特征
    df_pd["log_qty"] = np.log1p(df_pd["委托数量"])
    
    # 价格相关的对数变换（使用log1p处理可能的零值）
    df_pd["log_order_price"] = np.log1p(pd.to_numeric(df_pd["委托价格"], errors='coerce').fillna(0))
    df_pd["log_bid1"] = np.log1p(pd.to_numeric(df_pd.get("申买价1", 0), errors='coerce').fillna(0))
    df_pd["log_ask1"] = np.log1p(pd.to_numeric(df_pd.get("申卖价1", 0), errors='coerce').fillna(0))
    
    # 量相关的对数变换
    df_pd["log_bid_vol"] = np.log1p(pd.to_numeric(df_pd.get("申买量1", 0), errors='coerce').fillna(0))
    df_pd["log_ask_vol"] = np.log1p(pd.to_numeric(df_pd.get("申卖量1", 0), errors='coerce').fillna(0))
    
    # 金额的对数变换（委托金额 = 委托价格 × 委托数量）
    委托金额 = pd.to_numeric(df_pd["委托价格"], errors='coerce') * pd.to_numeric(df_pd["委托数量"], errors='coerce')
    df_pd["log_order_amount"] = np.log1p(委托金额.fillna(0))

    # 确保时间排序
    df_pd.sort_values("委托_datetime", inplace=True)
    
    # 严格防泄露：滚动窗口仅使用委托时刻之前的数据
    # 注意：这里统计的是所有委托事件，包括当前委托之前的其他委托
    df_pd["orders_100ms"] = (df_pd.rolling("100ms", on="委托_datetime", closed='left')["委托_datetime"].count().fillna(0))
    df_pd["orders_1s"] = (df_pd.rolling("1s", on="委托_datetime", closed='left')["委托_datetime"].count().fillna(0))
    
    # 注意：移除基于"事件类型"的特征计算，因为这属于未来信息
    # 改为使用委托时刻的可观测信息，如价格位置等
    
    # 简化统计：只统计订单总数，不区分成交撤单（避免使用未来信息）
    # 如果需要撤单统计，应该使用其他委托的历史撤单信息，而不是当前委托的事件类型
    
    # 价格位置特征（可观测）
    df_pd["at_bid"] = (df_pd["委托价格"] <= df_pd["bid1"]).astype(int)
    df_pd["at_ask"] = (df_pd["委托价格"] >= df_pd["ask1"]).astype(int)
    df_pd["inside_spread"] = ((df_pd["委托价格"] > df_pd["bid1"]) & 
                             (df_pd["委托价格"] < df_pd["ask1"])).astype(int)

    # Ensure '委托_datetime' is datetime
    df_pd['委托_datetime'] = pd.to_datetime(df_pd['委托_datetime'])
    sec = (df_pd["委托_datetime"] - df_pd["委托_datetime"].dt.normalize() - pd.Timedelta("09:30:00")).dt.total_seconds()
    df_pd["time_sin"] = np.sin(2*np.pi*sec/(4.5*3600))
    df_pd["time_cos"] = np.cos(2*np.pi*sec/(4.5*3600))

    df_pd["in_auction"] = (((df_pd["委托_datetime"].dt.time >= pd.to_datetime("09:15").time()) & (df_pd["委托_datetime"].dt.time < pd.to_datetime("09:30").time())) |
                         ((df_pd["委托_datetime"].dt.time >= pd.to_datetime("14:57").time()) & (df_pd["委托_datetime"].dt.time <= pd.to_datetime("15:00").time()))).astype(int)
    
    # 修正价格偏离度：转换为bps
    safe_prev_close = df_pd["prev_close"].replace(0, np.nan)
    df_pd["price_dev_prevclose_bps"] = ((df_pd["委托价格"] - df_pd["prev_close"]) / safe_prev_close * 10000).fillna(0)
    
    # 价格激进度（实时可计算）
    price_aggressiveness = np.where(
        df_pd["is_buy"] == 1,
        (df_pd["委托价格"] - df_pd["bid1"]) / (df_pd["spread"] + 1e-8),
        (df_pd["ask1"] - df_pd["委托价格"]) / (df_pd["spread"] + 1e-8)
    )
    df_pd["price_aggressiveness"] = pd.Series(price_aggressiveness).fillna(0)
    
    # 新增：挂单簇拥度（实时版本）
    price_distance = np.abs(df_pd["委托价格"] - df_pd["mid_price"])
    df_pd["cluster_score"] = 1.0 / (price_distance + 1e-6)  # 距离越近，分数越高
    
    return df_pd

def calculate_enhanced_realtime_features(df_pd: pd.DataFrame) -> pd.DataFrame:
    """计算扩展的实时特征"""
    try:
        # 确保按ticker和时间排序
        df_pd = df_pd.sort_values(['ticker', '委托_datetime'])
        
        # 新增：异常生存时间（相对于当日该股票的历史均值标准化）
        # 使用当前时刻之前的历史数据计算均值和标准差
        def calc_rolling_zscore(group):
            # 计算存活时间（对于委托事件，用当前时间减去委托时间的秒数作为代理）
            current_time = group['委托_datetime']
            group['temp_survival'] = (current_time - current_time.shift(1)).dt.total_seconds().fillna(0) * 1000
            
            # 滚动计算z-score（使用expanding window，仅用历史数据）
            rolling_mean = group['temp_survival'].expanding(min_periods=1).mean().shift(1).fillna(0)
            rolling_std = group['temp_survival'].expanding(min_periods=1).std().shift(1).fillna(1)
            group['z_survival'] = (group['temp_survival'] - rolling_mean) / (rolling_std + 1e-8)
            
            return group
        
        df_pd = df_pd.groupby('ticker').apply(calc_rolling_zscore).reset_index(drop=True)
        
        # 新增：短期价格动量（使用历史数据）
        df_pd['mid_price_lag1'] = df_pd.groupby('ticker')['mid_price'].shift(1)
        df_pd['price_momentum_100ms'] = (df_pd['mid_price'] - df_pd['mid_price_lag1']) / (df_pd['mid_price_lag1'] + 1e-8)
        
        # 新增：短期价差变化
        df_pd['spread_lag1'] = df_pd.groupby('ticker')['spread'].shift(1)
        df_pd['spread_change'] = (df_pd['spread'] - df_pd['spread_lag1']) / (df_pd['spread_lag1'] + 1e-8)
        
        # 新增：订单密度指标
        df_pd['order_density'] = df_pd['orders_100ms'] / 1.0  # 每100ms的订单数
        
        # 清理临时列
        df_pd = df_pd.drop(['temp_survival', 'mid_price_lag1', 'spread_lag1'], axis=1, errors='ignore')
        
    except Exception as e:
        console.print(f"  ⚠️ Error in extended realtime features calculation: {e}")
        # 如果计算失败，填充默认值
        for col in ['z_survival', 'price_momentum_100ms', 'spread_change', 'order_density']:
            if col not in df_pd.columns:
                df_pd[col] = 0.0
    
    return df_pd

def calculate_order_book_pressure(df_pd: pd.DataFrame) -> pd.DataFrame:
    """计算订单簿压力指标 (Pandas version)"""
    try:
        # 优先使用 bid_vol1/ask_vol1，如果不存在则使用原始列
        if 'bid_vol1' in df_pd.columns and 'ask_vol1' in df_pd.columns:
            df_pd['book_imbalance'] = (df_pd['bid_vol1'] - df_pd['ask_vol1']) / (df_pd['bid_vol1'] + df_pd['ask_vol1'] + 1e-8)
        elif '申买量1' in df_pd.columns and '申卖量1' in df_pd.columns:
            df_pd['book_imbalance'] = (df_pd['申买量1'] - df_pd['申卖量1']) / (df_pd['申买量1'] + df_pd['申卖量1'] + 1e-8)
        else:
            df_pd['book_imbalance'] = 0.0

        required_cols = ['委托价格', 'bid1', 'ask1', 'is_buy']
        # Ensure 'bid1', 'ask1', 'is_buy' are present from calc_realtime_features
        if 'bid1' not in df_pd.columns and '申买价1' in df_pd.columns: 
            df_pd['bid1'] = df_pd['申买价1']
        if 'ask1' not in df_pd.columns and '申卖价1' in df_pd.columns: 
            df_pd['ask1'] = df_pd['申卖价1']
        if 'is_buy' not in df_pd.columns and '方向_委托' in df_pd.columns: 
            df_pd['is_buy'] = (df_pd['方向_委托'] == "买").astype(int)

        missing_cols = [col for col in required_cols if col not in df_pd.columns]
        
        if missing_cols:
            df_pd['price_aggressiveness'] = 0.0
        else:
            if 'spread' not in df_pd.columns:
                 if 'bid1' in df_pd.columns and 'ask1' in df_pd.columns:
                    df_pd['spread'] = df_pd['ask1'] - df_pd['bid1']
                 else:
                    df_pd['spread'] = 0.0

            df_pd['price_aggressiveness'] = np.where(
                df_pd['is_buy'] == 1,
                (df_pd['委托价格'] - df_pd['bid1']) / (df_pd['spread'].replace(0, 1e-8) + 1e-8),
                (df_pd['ask1'] - df_pd['委托价格']) / (df_pd['spread'].replace(0, 1e-8) + 1e-8)
            ).astype(float)
            df_pd['price_aggressiveness'].fillna(0.0, inplace=True)

    except Exception as e:
        df_pd['book_imbalance'] = df_pd.get('book_imbalance', 0.0)
        df_pd['price_aggressiveness'] = df_pd.get('price_aggressiveness', 0.0)
    
    return df_pd

# ──────────────────────────── Polars Feature Engineering ────────────────────────────
def calc_realtime_features_polars(df: pl.LazyFrame) -> pl.LazyFrame:
    """使用 Polars Lazy API 计算实时可观测特征（修正版）"""
    df = df.with_columns([
        pl.col("委托_datetime").cast(pl.Datetime("ns")).alias("ts_ns"),
        pl.col("事件_datetime").cast(pl.Datetime("ns")).alias("event_ts_ns"),
    ])
    
    # 基础盘口特征
    df = df.with_columns([
        pl.col("申买价1").alias("bid1").cast(pl.Float64),
        pl.col("申卖价1").alias("ask1").cast(pl.Float64),
        pl.col("前收盘").alias("prev_close").cast(pl.Float64),
        pl.col("申买量1").alias("bid_vol1").cast(pl.Float64).fill_null(0),
        pl.col("申卖量1").alias("ask_vol1").cast(pl.Float64).fill_null(0),
        (pl.col("方向_委托") == "买").cast(pl.Int8).alias("is_buy"),
        # 移除基于"事件类型"的特征，因为这属于未来信息
        # 对数变换：处理尺度过大的特征
        pl.col("委托数量").log1p().alias("log_qty"),
        pl.col("委托价格").cast(pl.Float64).log1p().alias("log_order_price"),
        pl.col("申买价1").cast(pl.Float64).log1p().alias("log_bid1"),
        pl.col("申卖价1").cast(pl.Float64).log1p().alias("log_ask1"),
        pl.col("申买量1").cast(pl.Float64).log1p().alias("log_bid_vol"),
        pl.col("申卖量1").cast(pl.Float64).log1p().alias("log_ask_vol"),
        pl.col("委托价格").cast(pl.Float64),
    ])
    
    # 盘口衍生特征
    df = df.with_columns([
        ((pl.col("bid1") + pl.col("ask1")) / 2).alias("mid_price"),
        (pl.col("ask1") - pl.col("bid1")).alias("spread"),
    ])
    
    # 价格相关特征（修正版）
    df = df.with_columns([
        (pl.col("委托价格") - pl.col("mid_price")).alias("delta_mid"),
        # 修正pct_spread：防止溢出
        ((pl.col("委托价格") - pl.col("mid_price")).abs() / pl.col("mid_price") * 100).fill_null(0).clip(-500, 500).alias("pct_spread"),
        # 价格偏离度转换为bps
        ((pl.col("委托价格") - pl.col("prev_close")) / pl.col("prev_close") * 10000).fill_null(0).alias("price_dev_prevclose_bps"),
    ])
    
    # 价格激进度（实时版）
    df = df.with_columns([
        pl.when(pl.col("is_buy") == 1)
        .then((pl.col("委托价格") - pl.col("bid1")) / (pl.col("spread") + 1e-8))
        .otherwise((pl.col("ask1") - pl.col("委托价格")) / (pl.col("spread") + 1e-8))
        .fill_null(0).alias("price_aggressiveness")
    ])
    
    # 挂单簇拥度
    df = df.with_columns([
        (1.0 / ((pl.col("委托价格") - pl.col("mid_price")).abs() + 1e-6)).alias("cluster_score"),
        # 对数金额
        (pl.col("委托价格") * pl.col("委托数量")).log1p().alias("log_order_amount")
    ])
    
    # 时间特征
    df = df.with_columns([
        ((pl.col("委托_datetime").dt.hour() - 9) * 3600 + 
         (pl.col("委托_datetime").dt.minute() - 30) * 60 + 
         pl.col("委托_datetime").dt.second()).alias("seconds_since_market_open"),
        (pl.col("委托_datetime").dt.time().is_between(pl.time(9,15), pl.time(9,30), "left") |
         pl.col("委托_datetime").dt.time().is_between(pl.time(14,57), pl.time(15,0), "both")
        ).cast(pl.Int8).alias("in_auction"),
    ])
    
    df = df.with_columns([
        (2 * np.pi * pl.col("seconds_since_market_open") / (4.5 * 3600)).sin().alias("time_sin"),
        (2 * np.pi * pl.col("seconds_since_market_open") / (4.5 * 3600)).cos().alias("time_cos"),
    ])
    
    # 滚动窗口特征（修正版：使用closed='left'避免未来信息）
    try:
        df = df.with_columns([
            pl.col("ts_ns").set_sorted().alias("ts_ns_sorted")
        ])

        # 修正滚动窗口：只统计订单数，不使用未来信息
        orders_100ms_expr = pl.count().rolling(index_column="ts_ns_sorted", period="100ms", closed="left").fill_null(0)
        orders_1s_expr = pl.count().rolling(index_column="ts_ns_sorted", period="1s", closed="left").fill_null(0)
        
        df = df.with_columns([
            orders_100ms_expr.alias("orders_100ms"),
            orders_1s_expr.alias("orders_1s"),
        ]).drop("ts_ns_sorted")
        
        # 添加价格位置特征（可观测信息）
        df = df.with_columns([
            (pl.col("委托价格") <= pl.col("bid1")).cast(pl.Int8).alias("at_bid"),
            (pl.col("委托价格") >= pl.col("ask1")).cast(pl.Int8).alias("at_ask"),
            ((pl.col("委托价格") > pl.col("bid1")) & 
             (pl.col("委托价格") < pl.col("ask1"))).cast(pl.Int8).alias("inside_spread"),
        ])

    except Exception as e:
        console.print(f"[yellow]Warning: Polars rolling window calculation failed ({e}), using default values[/yellow]")
        df = df.with_columns([
            pl.lit(0).cast(pl.UInt32).alias("orders_100ms"),
            pl.lit(0).cast(pl.UInt32).alias("orders_1s"),
            pl.lit(0).cast(pl.Int8).alias("at_bid"),
            pl.lit(0).cast(pl.Int8).alias("at_ask"),
            pl.lit(0).cast(pl.Int8).alias("inside_spread"),
        ])
    return df

def calculate_order_book_pressure_polars(df: pl.LazyFrame, schema: dict) -> pl.LazyFrame:
    """使用 Polars 计算订单簿压力指标"""
    # 只在 book_imbalance 不存在时创建
    if 'book_imbalance' not in schema:
        # 优先使用 bid_vol1/ask_vol1，如果不存在则使用原始列
        if 'bid_vol1' in schema and 'ask_vol1' in schema:
            df = df.with_columns(
                ((pl.col('bid_vol1') - pl.col('ask_vol1')) / 
                 (pl.col('bid_vol1') + pl.col('ask_vol1') + 1e-8))
                .fill_null(0.0).alias('book_imbalance')
            )
        elif '申买量1' in schema and '申卖量1' in schema:
            df = df.with_columns(
                ((pl.col('申买量1').cast(pl.Float64) - pl.col('申卖量1').cast(pl.Float64)) / 
                 (pl.col('申买量1').cast(pl.Float64) + pl.col('申卖量1').cast(pl.Float64) + 1e-8))
                .fill_null(0.0).alias('book_imbalance')
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias('book_imbalance'))

    # 只在 price_aggressiveness 不存在时创建
    if 'price_aggressiveness' not in schema:
        pa_req_cols = ['委托价格', 'bid1', 'ask1', 'is_buy', 'spread']
        if all(c in schema for c in pa_req_cols):
             df = df.with_columns(
                pl.when(pl.col('is_buy') == 1)
                .then((pl.col('委托价格') - pl.col('bid1')) / (pl.col('spread').replace(0, 1e-8) + 1e-8))
                .otherwise((pl.col('ask1') - pl.col('委托价格')) / (pl.col('spread').replace(0, 1e-8) + 1e-8))
                .fill_null(0.0).alias('price_aggressiveness')
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias('price_aggressiveness'))
    return df 

# ──────────────────────────── Feature Selection Utilities ────────────────────────────

def get_ml_friendly_features() -> List[str]:
    """返回适合机器学习的特征列表（偏好对数变换后的特征）"""
    ml_features = [
        # 基础特征
        "ticker", "自然日", "交易所委托号",
        
        # 对数变换特征（优先使用）
        "log_qty", "log_order_price", "log_bid1", "log_ask1", 
        "log_bid_vol", "log_ask_vol", "log_order_amount",
        
        # 标准化特征
        "is_buy", "mid_price", "spread", "delta_mid", 
        "pct_spread", "price_aggressiveness", "book_imbalance",
        
        # 时间特征
        "time_sin", "time_cos", "in_auction", "seconds_since_market_open",
        
        # 滚动统计特征（移除基于未来信息的特征）
        "orders_100ms", "orders_1s",
        
        # 复合特征
        "cluster_score", "price_dev_prevclose_bps"
    ]
    return ml_features

def get_interpretable_features() -> List[str]:
    """返回易于解释的原始特征列表"""
    interpretable_features = [
        # 基础信息
        "ticker", "自然日", "交易所委托号", "事件类型", "方向_委托",
        
        # 原始价格数量（业务含义清晰）
        "委托价格", "委托数量", "申买价1", "申卖价1", "申买量1", "申卖量1", "前收盘",
        
        # 简单衍生特征
        "is_buy", "mid_price", "spread", "存活时间_ms", "委托金额"
    ]
    return interpretable_features

def get_feature_groups() -> Dict[str, List[str]]:
    """按类型分组特征"""
    return {
        "价格特征": ["委托价格", "申买价1", "申卖价1", "前收盘", "mid_price", "log_order_price", "log_bid1", "log_ask1"],
        "数量特征": ["委托数量", "申买量1", "申卖量1", "log_qty", "log_bid_vol", "log_ask_vol"],
        "金额特征": ["委托金额", "log_order_amount"],
        "时间特征": ["time_sin", "time_cos", "in_auction", "seconds_since_market_open", "存活时间_ms"],
        "技术特征": ["spread", "pct_spread", "delta_mid", "price_aggressiveness", "book_imbalance", "cluster_score"],
        "统计特征": ["orders_100ms", "orders_1s", "at_bid", "at_ask", "inside_spread"]
    } 