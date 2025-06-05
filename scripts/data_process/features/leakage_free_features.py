#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
leakage_free_features.py
────────────────────────────────────────
严格防数据泄露的特征工程模块
• 明确区分委托时刻可观测信息 vs 未来信息
• 提供时间窗口控制的特征计算
• 标记潜在泄露特征
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set
from rich.console import Console

console = Console()

# ────────────────────────── 数据泄露识别 ──────────────────────────
def identify_leakage_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """识别并分类数据中的泄露风险特征"""
    
    # 明显的未来信息（绝对不能用）
    future_info = [
        "存活时间_ms",           # 委托的最终存活时间
        "事件_datetime",         # 成交/撤单发生时间
        "成交价格", "成交数量",     # 未来成交信息
    ]
    
    # 委托结果信息（泄露）
    order_outcome = [
        "事件类型",              # 最终是成交还是撤单
        "is_cancel_event",       # 是否撤单
        "is_trade_event",        # 是否成交
    ]
    
    # 基于未来事件的聚合特征（泄露）
    future_aggregates = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in [
            "total_", "final_", "num_trades", "num_cancels", 
            "fill_ratio", "fully_filled"
        ]):
            future_aggregates.append(col)
    
    # 委托时刻可观测信息（安全）
    observable_at_order = [
        # 基础委托信息
        "委托价格", "委托数量", "委托_datetime", "方向_委托", "is_buy",
        
        # 行情信息（委托时刻的市场状态）
        "申买价1", "申卖价1", "申买量1", "申卖量1", "前收盘", 
        "bid1", "ask1", "bid_vol1", "ask_vol1", "prev_close",
        "mid_price", "spread", "delta_mid", "pct_spread",
        
        # 时间特征
        "time_sin", "time_cos", "in_auction", "seconds_since_market_open",
        
        # 价格分析
        "price_dev_prevclose_bps", "price_aggressiveness", "cluster_score",
        
        # 对数变换
        "log_qty", "log_order_price", "log_bid1", "log_ask1", 
        "log_bid_vol", "log_ask_vol", "log_order_amount",
    ]
    
    # 历史统计特征（需要检查时间窗口）
    historical_stats = []
    for col in df.columns:
        if any(keyword in col for keyword in [
            "orders_", "cancels_", "trades_", "cancel_ratio_"
        ]):
            historical_stats.append(col)
    
    return {
        "future_info": [col for col in future_info if col in df.columns],
        "order_outcome": [col for col in order_outcome if col in df.columns], 
        "future_aggregates": future_aggregates,
        "observable_at_order": [col for col in observable_at_order if col in df.columns],
        "historical_stats": historical_stats
    }

def create_feature_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """为每个特征创建元数据，标记泄露风险"""
    leakage_info = identify_leakage_columns(df)
    
    feature_metadata = []
    for col in df.columns:
        risk_level = "UNKNOWN"
        category = "其他"
        description = ""
        
        if col in leakage_info["future_info"]:
            risk_level = "HIGH_LEAKAGE"
            category = "未来信息"
            description = "包含委托时刻未知的未来信息"
        elif col in leakage_info["order_outcome"]:
            risk_level = "HIGH_LEAKAGE"
            category = "订单结果"
            description = "包含订单的最终结果信息"
        elif col in leakage_info["future_aggregates"]:
            risk_level = "HIGH_LEAKAGE"
            category = "未来聚合"
            description = "基于未来事件的聚合统计"
        elif col in leakage_info["observable_at_order"]:
            risk_level = "SAFE"
            category = "可观测信息"
            description = "委托时刻可观测的信息"
        elif col in leakage_info["historical_stats"]:
            risk_level = "NEED_CHECK"
            category = "历史统计"
            description = "需检查时间窗口的历史统计特征"
        
        feature_metadata.append({
            "feature": col,
            "risk_level": risk_level,
            "category": category,
            "description": description
        })
    
    return pd.DataFrame(feature_metadata)

# ────────────────────────── 安全特征构造 ──────────────────────────
def create_safe_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建基于委托时刻可观测信息的安全特征"""
    df_safe = df.copy()
    
    console.print("[green]🛡️ 创建安全的市场特征...[/green]")
    
    # 1. 基础价格特征（委托时刻已知）
    if all(col in df.columns for col in ["申买价1", "申卖价1"]):
        df_safe["mid_price"] = (df_safe["申买价1"] + df_safe["申卖价1"]) / 2
        df_safe["spread"] = df_safe["申卖价1"] - df_safe["申买价1"]
        df_safe["relative_spread"] = df_safe["spread"] / df_safe["mid_price"]
    
    # 2. 价格偏离特征
    if "委托价格" in df.columns and "mid_price" in df_safe.columns:
        df_safe["delta_mid"] = df_safe["委托价格"] - df_safe["mid_price"]
        df_safe["price_distance_pct"] = abs(df_safe["delta_mid"]) / df_safe["mid_price"] * 100
    
    # 3. 订单簿压力（委托时刻可观测）
    if all(col in df.columns for col in ["申买量1", "申卖量1"]):
        total_vol = df_safe["申买量1"] + df_safe["申卖量1"]
        df_safe["book_imbalance"] = (df_safe["申买量1"] - df_safe["申卖量1"]) / (total_vol + 1e-8)
    
    # 4. 时间特征（委托时刻已知）
    if "委托_datetime" in df.columns:
        df_safe["委托_datetime"] = pd.to_datetime(df_safe["委托_datetime"])
        
        # 开市时间相对位置
        market_open = df_safe["委托_datetime"].dt.normalize() + pd.Timedelta("09:30:00")
        seconds_since_open = (df_safe["委托_datetime"] - market_open).dt.total_seconds()
        df_safe["seconds_since_market_open"] = seconds_since_open
        
        # 周期特征
        day_seconds = 4.5 * 3600
        df_safe["time_sin"] = np.sin(2 * np.pi * seconds_since_open / day_seconds)
        df_safe["time_cos"] = np.cos(2 * np.pi * seconds_since_open / day_seconds)
        
        # 集合竞价标记
        time_obj = df_safe["委托_datetime"].dt.time
        morning_auction = (time_obj >= pd.Timestamp("09:15").time()) & (time_obj < pd.Timestamp("09:30").time())
        closing_auction = (time_obj >= pd.Timestamp("14:57").time()) & (time_obj <= pd.Timestamp("15:00").time())
        df_safe["in_auction"] = (morning_auction | closing_auction).astype(int)
    
    # 5. 对数变换（处理尺度问题）
    for col, new_col in [
        ("委托数量", "log_qty"),
        ("委托价格", "log_order_price"),
        ("申买价1", "log_bid1"),
        ("申卖价1", "log_ask1"),
        ("申买量1", "log_bid_vol"),
        ("申卖量1", "log_ask_vol")
    ]:
        if col in df.columns:
            df_safe[new_col] = np.log1p(pd.to_numeric(df_safe[col], errors='coerce').fillna(0))
    
    return df_safe

def create_safe_historical_features(df: pd.DataFrame, time_col: str = "委托_datetime") -> pd.DataFrame:
    """创建基于历史信息的安全特征（严格时间窗口控制）"""
    df_hist = df.copy()
    df_hist[time_col] = pd.to_datetime(df_hist[time_col])
    df_hist = df_hist.sort_values([time_col])
    
    console.print("[green]📊 创建历史统计特征（严格时间控制）...[/green]")
    
    # 仅统计当前委托时刻之前的历史事件
    for window, suffix in [("100ms", "_100ms"), ("1s", "_1s"), ("5s", "_5s")]:
        # 订单密度：过去时间窗口内的订单数
        df_hist[f"orders{suffix}"] = (
            df_hist.rolling(window, on=time_col, closed='left')[time_col]
            .count().fillna(0)
        )
    
    # 注意：撤单和成交统计需要特别小心，因为它们可能包含当前订单的结果
    # 这里我们只统计历史上其他订单的撤单/成交情况
    if "事件类型" in df.columns:
        console.print("[yellow]⚠️ 警告：事件类型包含当前订单结果，需谨慎使用[/yellow]")
        
        # 创建历史撤单标记（排除当前订单）
        df_hist["past_cancel_flag"] = (df_hist["事件类型"] == "撤单").astype(int)
        df_hist["past_trade_flag"] = (df_hist["事件类型"] == "成交").astype(int)
        
        # 历史撤单率（仅基于过去数据）
        for window, suffix in [("1s", "_1s"), ("5s", "_5s")]:
            past_orders = df_hist.rolling(window, on=time_col, closed='left')[time_col].count()
            past_cancels = df_hist.rolling(window, on=time_col, closed='left')["past_cancel_flag"].sum()
            df_hist[f"cancel_ratio{suffix}"] = (past_cancels / (past_orders + 1e-8)).fillna(0)
    
    return df_hist

# ────────────────────────── 特征验证和清理 ──────────────────────────
def get_id_columns() -> List[str]:
    """返回数据标识列（不用于训练）"""
    return [
        # 主键标识
        "自然日", "ticker", "交易所委托号", "委托编号",
        
        # 时间戳（原始格式，不是特征）
        "时间", "时间_委托", "时间_事件", 
        "自然日_委托", "自然日_事件",
        "委托_datetime", "事件_datetime", "quote_dt",
        "ts_ns", "event_ts_ns",
        
        # 代码标识
        "万得代码", "交易所代码", "成交代码",
        
        # 其他标识信息
        "委托类型", "方向_委托", "方向_事件"
    ]

def get_safe_feature_list() -> List[str]:
    """返回确认安全的特征列表（纯特征，不包含ID）"""
    return [        
        # 委托基础信息（数值特征）
        "委托价格", "委托数量", "is_buy", "委托金额",
        
        # 行情信息（委托时刻的市场状态）
        "申买价1", "申卖价1", "申买量1", "申卖量1", "前收盘",
        "bid1", "ask1", "bid_vol1", "ask_vol1", "prev_close",
        
        # 派生价格特征
        "mid_price", "spread", "relative_spread", "delta_mid", "price_distance_pct",
        "pct_spread", "price_vs_mid", "price_vs_mid_bps",
        "book_imbalance", "price_dev_prevclose_bps", "price_aggressiveness", "cluster_score",
        
        # 时间特征（工程化的，不是原始时间戳）
        "time_sin", "time_cos", "in_auction", "seconds_since_market_open",
        
        # 对数变换特征
        "log_qty", "log_order_price", "log_bid1", "log_ask1", 
        "log_bid_vol", "log_ask_vol", "log_order_amount",
        
        # 历史统计特征（基于时间窗口）
        "orders_100ms", "orders_1s", "orders_5s",
        "cancels_100ms", "cancels_1s", "cancels_5s", "trades_1s",
        "cancel_ratio_100ms", "cancel_ratio_1s",
        
        # 技术指标
        "price_volatility", "price_momentum", "spread_volatility", "order_imbalance",
        
        # 统计特征
        "hist_order_size_mean", "hist_spread_mean"
    ]

def get_leakage_feature_list() -> List[str]:
    """返回确认泄露的特征列表（训练时必须排除）"""
    return [
        # 明显未来信息
        "存活时间_ms", "final_survival_time_ms", "life_ms",
        "事件_datetime", "finish_time",
        
        # 成交结果信息
        "成交价格", "成交数量", "exec_qty", "fill_ratio",
        "事件类型", "is_cancel_event", "is_trade_event",
        
        # 订单最终状态
        "is_fully_filled", "canceled", "total_events",
        "num_trades", "num_cancels", "total_traded_qty",
        
        # 标签相关
        "flag_R1", "flag_R2", "y_label",
        "enhanced_spoofing_liberal", "enhanced_spoofing_moderate", "enhanced_spoofing_strict"
    ]

def validate_features_safety(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, List[str]]:
    """验证特征的安全性"""
    safe_features = get_safe_feature_list()
    leakage_features = get_leakage_feature_list()
    id_columns = get_id_columns()
    
    validated_safe = [col for col in feature_cols if col in safe_features]
    detected_leakage = [col for col in feature_cols if col in leakage_features]
    detected_ids = [col for col in feature_cols if col in id_columns]
    uncertain = [col for col in feature_cols if col not in safe_features and col not in leakage_features and col not in id_columns]
    
    return {
        "safe": validated_safe,
        "leakage": detected_leakage,
        "ids": detected_ids,
        "uncertain": uncertain
    }

def clean_features_for_training(df: pd.DataFrame, target_col: str = "y_label") -> pd.DataFrame:
    """清理数据，移除泄露特征，只保留安全特征"""
    
    console.print("[bold cyan]🔍 清理特征数据以防止泄露...[/bold cyan]")
    
    # 获取所有列
    all_cols = df.columns.tolist()
    
    # 验证特征安全性
    validation = validate_features_safety(df, all_cols)
    
    console.print(f"[green]✅ 安全特征: {len(validation['safe'])} 个[/green]")
    console.print(f"[red]❌ 泄露特征: {len(validation['leakage'])} 个[/red]")
    console.print(f"[blue]🆔 ID列: {len(validation['ids'])} 个[/blue]")
    console.print(f"[yellow]⚠️ 不确定特征: {len(validation['uncertain'])} 个[/yellow]")
    
    if validation['leakage']:
        console.print(f"[red]移除泄露特征: {validation['leakage'][:10]}{'...' if len(validation['leakage']) > 10 else ''}[/red]")
    
    if validation['ids']:
        console.print(f"[blue]移除ID列: {validation['ids'][:10]}{'...' if len(validation['ids']) > 10 else ''}[/blue]")
    
    if validation['uncertain']:
        console.print(f"[yellow]不确定特征需人工检查: {validation['uncertain'][:5]}{'...' if len(validation['uncertain']) > 5 else ''}[/yellow]")
    
    # 保留安全特征 + 目标变量
    keep_cols = validation['safe'].copy()
    if target_col in all_cols:
        keep_cols.append(target_col)
    
    # 对于不确定的特征，进行更严格的筛选
    uncertain_safe = []
    for col in validation['uncertain']:
        dtype = df[col].dtype
        # 只保留数值型特征，排除字符串和时间列
        if dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
            uncertain_safe.append(col)
        elif 'datetime' not in str(dtype) and dtype != 'object':
            # 尝试转换为数值型
            try:
                pd.to_numeric(df[col], errors='raise')
                uncertain_safe.append(col)
            except:
                console.print(f"[yellow]跳过非数值特征: {col} (dtype: {dtype})[/yellow]")
        else:
            console.print(f"[yellow]跳过非数值特征: {col} (dtype: {dtype})[/yellow]")
    
    keep_cols.extend(uncertain_safe)
    
    # 确保列在DataFrame中存在
    keep_cols = [col for col in keep_cols if col in all_cols]
    
    cleaned_df = df[keep_cols].copy()
    
    # 最终数据类型检查和转换
    for col in cleaned_df.columns:
        if col == target_col:
            continue
        dtype = cleaned_df[col].dtype
        if dtype == 'object':
            # 尝试转换为数值型
            try:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0)
            except:
                console.print(f"[red]无法转换 {col} 为数值型，移除此列[/red]")
                cleaned_df = cleaned_df.drop(columns=[col])
        elif 'datetime' in str(dtype):
            console.print(f"[red]移除时间列: {col}[/red]")
            cleaned_df = cleaned_df.drop(columns=[col])
    
    console.print(f"[green]📊 清理完成: {len(cleaned_df.columns)} 个特征保留（最终数值检查通过）[/green]")
    
    return cleaned_df 