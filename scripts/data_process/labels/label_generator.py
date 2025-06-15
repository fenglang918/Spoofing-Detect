#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
label_generator.py
────────────────────────────────────────
标签生成模块 - 第二阶段标签计算
• 从委托事件流生成欺骗标签
• 支持多种标签规则（R1, R2, 扩展规则）
• 批量处理多日期数据
• 模块化标签计算流水线

重要修复说明:
────────────────────────────────────────
v2.1 - 修复开盘时间误标问题:
• 扩展规则现在排除开盘时间段(09:30-10:00, 13:00-13:15)的正常大单
• 异常时间检测改为集合竞价晚期(09:25-09:30)和午休时间
• 激进定价在开盘时间需要更极端的价格偏离才触发
• 综合标签策略更保守，避免将正常开盘活动误标为欺诈
"""

import sys
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Set, List, Optional, Union, Dict
import argparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

console = Console()

# ────────────────────────────── 列定义 ──────────────────────────────

# 主键列（用于数据合并和标识）
KEY_COLUMNS = [
    "自然日", 
    "ticker", 
    "交易所委托号"
]

# 时间列（不用于训练，但可能用于分析）
TIME_COLUMNS = [
    "委托_datetime", 
    "事件_datetime"
]

# 元数据列（保存但不用于训练）
METADATA_COLUMNS = [
    "委托数量", 
    "方向_委托", 
    "委托价格",
    "事件类型",
    "存活时间_ms"
]

# 基础标签列（由基础规则生成）
BASE_LABEL_COLUMNS = [
    "flag_R1",           # R1规则：快速撤单
    "flag_R2",           # R2规则：价格操纵  
    "y_label"            # 基础综合标签 (R1 & R2)
]

# 扩展标签列（由扩展规则生成）
ENHANCED_LABEL_COLUMNS = [
    "extreme_price_deviation",      # 极端价格偏离
    "aggressive_pricing",           # 激进定价
    "abnormal_large_order",         # 异常大单
    "volatile_period_anomaly",      # 异常时间段活动
    "enhanced_spoofing_conservative", # 保守版扩展标签
    "enhanced_spoofing_moderate",     # 中等版扩展标签  
    "enhanced_spoofing_liberal",      # 宽松版扩展标签
    "enhanced_spoofing_strict",       # 严格版扩展标签
    "enhanced_combined"               # 综合标签
]

# 训练用的标签列（主要目标变量）
TRAINING_TARGET_COLUMNS = [
    "y_label",                        # 基础标签
    "enhanced_spoofing_conservative", # 保守扩展标签
    "enhanced_spoofing_moderate",     # 中等扩展标签
    "enhanced_combined"               # 综合标签
]

# 所有标签列
ALL_LABEL_COLUMNS = BASE_LABEL_COLUMNS + ENHANCED_LABEL_COLUMNS

# 输出时需要保存的列（用于标签文件）
LABEL_OUTPUT_COLUMNS = KEY_COLUMNS + TIME_COLUMNS + METADATA_COLUMNS + ALL_LABEL_COLUMNS

def get_label_columns(include_enhanced: bool = True) -> List[str]:
    """
    获取标签列列表
    
    Args:
        include_enhanced: 是否包含扩展标签
        
    Returns:
        标签列列表
    """
    if include_enhanced:
        return BASE_LABEL_COLUMNS + ENHANCED_LABEL_COLUMNS
    else:
        return BASE_LABEL_COLUMNS

def get_training_target_columns() -> List[str]:
    """获取可用于训练的目标变量列表"""
    return TRAINING_TARGET_COLUMNS.copy()

def get_key_columns() -> List[str]:
    """获取主键列列表"""
    return KEY_COLUMNS.copy()

def get_metadata_columns() -> List[str]:
    """获取元数据列列表"""
    return METADATA_COLUMNS.copy()

def get_time_columns() -> List[str]:
    """获取时间列列表"""
    return TIME_COLUMNS.copy()

def get_label_output_columns(include_enhanced: bool = True) -> List[str]:
    """
    获取标签文件输出列列表
    
    Args:
        include_enhanced: 是否包含扩展标签
        
    Returns:
        输出列列表
    """
    columns = KEY_COLUMNS + TIME_COLUMNS + METADATA_COLUMNS
    columns.extend(get_label_columns(include_enhanced))
    return columns

# ────────────────────────────── 标签生成核心函数 ──────────────────────────────

def apply_basic_spoofing_rules_pandas(df: pd.DataFrame, r1_ms: int, r2_ms: int, r2_mult: float) -> pd.DataFrame:
    """
    基础欺骗标签规则 (Pandas版本)
    
    Args:
        df: 输入DataFrame
        r1_ms: R1规则时间阈值(毫秒)
        r2_ms: R2规则时间阈值(毫秒)
        r2_mult: R2规则价差倍数
        
    Returns:
        添加标签的DataFrame
    """
    try:
        # R1规则: 快速撤单
        df['flag_R1'] = (df['存活时间_ms'] < r1_ms) & (df['事件类型'] == '撤单')
        
        # R2规则: 价格操纵 
        if 'spread' in df.columns and 'delta_mid' in df.columns:
            safe_spread = df['spread'].fillna(np.inf).replace(0, np.inf)
            df['flag_R2'] = ((df['存活时间_ms'] < r2_ms) & 
                            (df['delta_mid'].abs() >= r2_mult * safe_spread))
            df['y_label'] = (df['flag_R1'] & df['flag_R2']).astype(int)
        else:
            df['flag_R2'] = False
            df['y_label'] = df['flag_R1'].astype(int)
            
        return df
        
    except Exception as e:
        console.print(f"[red]警告: 基础标签规则应用失败: {e}[/red]")
        if 'y_label' not in df.columns:
            df['y_label'] = 0
        return df

def apply_basic_spoofing_rules_polars(df: Union[pl.DataFrame, pl.LazyFrame], r1_ms: int, r2_ms: int, r2_mult: float) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    基础欺骗标签规则 (Polars版本)
    
    Args:
        df: 输入DataFrame/LazyFrame
        r1_ms: R1规则时间阈值(毫秒)
        r2_ms: R2规则时间阈值(毫秒) 
        r2_mult: R2规则价差倍数
        
    Returns:
        添加标签的DataFrame/LazyFrame
    """
    try:
        # 转换为LazyFrame
        if isinstance(df, pl.DataFrame):
            lazy_df = df.lazy()
            return_dataframe = True
        else:
            lazy_df = df
            return_dataframe = False
        
        # 获取列名
        schema = lazy_df.schema
        
        # R1规则: 快速撤单
        if all(col in schema for col in ['存活时间_ms', '事件类型']):
            lazy_df = lazy_df.with_columns([
                ((pl.col('存活时间_ms') < r1_ms) & (pl.col('事件类型') == '撤单')).alias('flag_R1')
            ])
        else:
            lazy_df = lazy_df.with_columns([pl.lit(False).alias('flag_R1')])
        
        # R2规则: 价格操纵
        if all(col in schema for col in ['存活时间_ms', 'spread', 'delta_mid']):
            safe_spread = pl.col('spread').fill_null(float('inf')).replace(0, float('inf'))
            lazy_df = lazy_df.with_columns([
                ((pl.col('存活时间_ms') < r2_ms) & 
                 (pl.col('delta_mid').abs() >= r2_mult * safe_spread)).alias('flag_R2')
            ])
            lazy_df = lazy_df.with_columns([
                (pl.col('flag_R1') & pl.col('flag_R2')).cast(pl.Int8).alias('y_label')
            ])
        else:
            lazy_df = lazy_df.with_columns([
                pl.lit(False).alias('flag_R2'),
                pl.col('flag_R1').cast(pl.Int8).alias('y_label')
            ])
        
        # 返回原始类型
        if return_dataframe:
            return lazy_df.collect()
        else:
            return lazy_df
            
    except Exception as e:
        console.print(f"[red]警告: Polars标签规则应用失败: {e}[/red]")
        # 如果失败，返回原始数据并添加基本标签
        if isinstance(df, pl.DataFrame):
            return df.with_columns(pl.lit(0).cast(pl.Int8).alias('y_label'))
        else:
            return df.with_columns(pl.lit(0).cast(pl.Int8).alias('y_label'))

def apply_enhanced_spoofing_rules_pandas(df: pd.DataFrame, r1_ms: int, r2_ms: int, r2_mult: float) -> pd.DataFrame:
    """
    扩展欺骗标签规则 (Pandas版本)
    
    Args:
        df: 输入DataFrame
        r1_ms: R1规则时间阈值(毫秒)
        r2_ms: R2规则时间阈值(毫秒)
        r2_mult: R2规则价差倍数
        
    Returns:
        添加扩展标签的DataFrame
    """
    try:
        # 先应用基础规则
        df = apply_basic_spoofing_rules_pandas(df, r1_ms, r2_ms, r2_mult)
        
        # 扩展规则1: 极端价格偏离
        if all(col in df.columns for col in ['委托价格', '委托数量', 'ticker']):
            df['extreme_price_deviation'] = (
                (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.9))
            ).astype(int)
        else:
            df['extreme_price_deviation'] = 0
        
        # 扩展规则2: 激进定价 (修复：在开盘时间段更保守)
        if all(col in df.columns for col in ['委托价格', 'bid1', 'ask1', '方向_委托', '委托_datetime']):
            df['委托_datetime'] = pd.to_datetime(df['委托_datetime'])
            
            # 开盘时间段
            opening_periods = (
                ((df['委托_datetime'].dt.time >= pd.to_datetime('09:30').time()) & 
                 (df['委托_datetime'].dt.time <= pd.to_datetime('10:00').time())) |
                ((df['委托_datetime'].dt.time >= pd.to_datetime('13:00').time()) & 
                 (df['委托_datetime'].dt.time <= pd.to_datetime('13:15').time()))
            )
            
            # 基础激进定价
            basic_aggressive = (
                ((df['方向_委托'] == '买') & (df['委托价格'] > df['ask1'])) |
                ((df['方向_委托'] == '卖') & (df['委托价格'] < df['bid1']))
            )
            
            # 开盘时间段需要更极端的价格偏离才标记
            spread = df['ask1'] - df['bid1']
            extreme_aggressive = (
                ((df['方向_委托'] == '买') & (df['委托价格'] > df['ask1'] + spread * 0.5)) |
                ((df['方向_委托'] == '卖') & (df['委托价格'] < df['bid1'] - spread * 0.5))
            )
            
            df['aggressive_pricing'] = (
                (opening_periods & extreme_aggressive) |  # 开盘时需要极端偏离
                (~opening_periods & basic_aggressive)     # 非开盘时正常标准
            ).astype(int)
        elif all(col in df.columns for col in ['委托价格', 'bid1', 'ask1', '方向_委托']):
            # 降级处理：没有时间信息时使用基础逻辑
            df['aggressive_pricing'] = (
                (((df['方向_委托'] == '买') & (df['委托价格'] > df['ask1'])) |
                 ((df['方向_委托'] == '卖') & (df['委托价格'] < df['bid1'])))
            ).astype(int)
        else:
            df['aggressive_pricing'] = 0
        
        # 扩展规则3: 异常大单 (修复：排除开盘时间段的正常大单)
        if all(col in df.columns for col in ['委托数量', 'ticker', '委托_datetime']):
            # 识别开盘时间段（正常交易活跃期）
            df['委托_datetime'] = pd.to_datetime(df['委托_datetime']) 
            opening_periods = (
                # 开盘前30分钟
                ((df['委托_datetime'].dt.time >= pd.to_datetime('09:30').time()) & 
                 (df['委托_datetime'].dt.time <= pd.to_datetime('10:00').time())) |
                # 午盘开盘前15分钟  
                ((df['委托_datetime'].dt.time >= pd.to_datetime('13:00').time()) & 
                 (df['委托_datetime'].dt.time <= pd.to_datetime('13:15').time()))
            )
            
            # 非开盘时间段的异常大单才标记
            df['abnormal_large_order'] = (
                (~opening_periods) &  # 排除开盘时间段
                (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.95))
            ).astype(int)
        elif all(col in df.columns for col in ['委托数量', 'ticker']):
            # 降级处理：没有时间信息时使用更高阈值
            df['abnormal_large_order'] = (
                df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.98)
            ).astype(int)
        else:
            df['abnormal_large_order'] = 0
        
        # 扩展规则4: 非正常交易时间异常活动 (修复：排除正常开盘收盘时间)
        if '委托_datetime' in df.columns:
            df['委托_datetime'] = pd.to_datetime(df['委托_datetime'])
            # 改为检测非正常时间的异常活动
            # 正常交易时间: 09:30-11:30, 13:00-15:00
            # 异常时间: 盘前集合竞价晚期 09:25-09:30, 午休时间等
            abnormal_time = (
                # 集合竞价晚期异常活动
                ((df['委托_datetime'].dt.time >= pd.to_datetime('09:25').time()) & 
                 (df['委托_datetime'].dt.time < pd.to_datetime('09:30').time())) |
                # 午休时间异常活动  
                ((df['委托_datetime'].dt.time >= pd.to_datetime('11:30').time()) & 
                 (df['委托_datetime'].dt.time < pd.to_datetime('13:00').time()))
            )
            
            if '委托数量' in df.columns and 'ticker' in df.columns:
                # 在异常时间段的大单才标记为可疑
                df['volatile_period_anomaly'] = (
                    abnormal_time &
                    (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.9))
                ).astype(int)
            else:
                df['volatile_period_anomaly'] = 0
        else:
            df['volatile_period_anomaly'] = 0
        
        # 综合扩展标签 (修复：更保守的标签策略)
        pattern_rules = ['extreme_price_deviation', 'aggressive_pricing', 'abnormal_large_order', 'volatile_period_anomaly']
        available_rules = [col for col in pattern_rules if col in df.columns]
        
        if available_rules:
            # 保守版本：至少两个规则触发 (避免误标)
            df['enhanced_spoofing_conservative'] = (df[available_rules].sum(axis=1) >= 2).astype(int)
            # 中等版本：多个规则触发
            df['enhanced_spoofing_moderate'] = (df[available_rules].sum(axis=1) >= 2).astype(int)
            # 宽松版本：任意规则+基础规则，或多个规则
            if 'y_label' in df.columns:
                df['enhanced_spoofing_liberal'] = (
                    (df['y_label'] == 1) | (df[available_rules].sum(axis=1) >= 2)
                ).astype(int)
            else:
                df['enhanced_spoofing_liberal'] = (df[available_rules].sum(axis=1) >= 1).astype(int)
            
            # 严格版本：大部分规则触发
            df['enhanced_spoofing_strict'] = (df[available_rules].sum(axis=1) >= 3).astype(int)
            
            # 结合原始规则的综合标签 (更保守)
            if 'y_label' in df.columns:
                df['enhanced_combined'] = (
                    (df['y_label'] == 1) | (df['enhanced_spoofing_conservative'] == 1)
                ).astype(int)
            else:
                df['enhanced_combined'] = df['enhanced_spoofing_conservative']
        
        return df
        
    except Exception as e:
        console.print(f"[red]警告: 扩展标签规则应用失败: {e}[/red]")
        return df

# ────────────────────────────── 标签生成器类 ──────────────────────────────

class LabelGenerator:
    """标签生成器"""
    
    def __init__(self, 
                 r1_ms: int = 50, 
                 r2_ms: int = 1000, 
                 r2_mult: float = 4.0,
                 extended_rules: bool = False,
                 backend: str = "polars"):
        """
        初始化标签生成器
        
        Args:
            r1_ms: R1规则时间阈值(ms)
            r2_ms: R2规则时间阈值(ms) 
            r2_mult: R2规则价差倍数
            extended_rules: 是否使用扩展标签规则
            backend: 计算后端
        """
        self.r1_ms = r1_ms
        self.r2_ms = r2_ms
        self.r2_mult = r2_mult
        self.extended_rules = extended_rules
        self.backend = backend
        self.console = Console()
        
        # 定义标签生成流水线
        self.label_pipeline = self._build_label_pipeline()
    
    def _build_label_pipeline(self) -> List[dict]:
        """构建标签生成流水线"""
        if self.extended_rules:
            pipeline = [
                {
                    "name": "扩展标签规则",
                    "function": "apply_enhanced_spoofing_rules",
                    "description": f"R1({self.r1_ms}ms) + R2({self.r2_ms}ms, {self.r2_mult}x) + 扩展规则"
                }
            ]
        else:
            pipeline = [
                {
                    "name": "基础标签规则", 
                    "function": "apply_basic_spoofing_rules",
                    "description": f"R1({self.r1_ms}ms) + R2({self.r2_ms}ms, {self.r2_mult}x)"
                }
            ]
        
        return pipeline
    
    def generate_labels_for_data(self,
                                data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
                                tickers: Optional[Set[str]] = None,
                                show_progress: bool = True) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        为给定数据生成标签
        
        Args:
            data: 输入的委托事件流数据
            tickers: 股票代码筛选
            show_progress: 是否显示进度条
            
        Returns:
            带标签的DataFrame
        """
        
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console,
                transient=True
            )
            progress.start()
        
        try:
            # 数据预处理
            if show_progress:
                prep_task = progress.add_task("预处理数据...", total=100)
            
            if self.backend == "polars":
                df_processed = self._preprocess_polars(data, tickers)
            else:
                df_processed = self._preprocess_pandas(data, tickers)
            
            if show_progress:
                progress.update(prep_task, advance=100)
            
            # 检查数据是否为空
            if self.backend == "polars":
                if isinstance(df_processed, pl.LazyFrame):
                    data_size = df_processed.collect().height
                else:
                    data_size = df_processed.height
            else:
                data_size = len(df_processed)
            
            if data_size == 0:
                self.console.print("[yellow]警告: 预处理后数据为空[/yellow]")
                return df_processed
            
            # 标签生成流水线
            for i, step in enumerate(self.label_pipeline):
                if show_progress:
                    label_task = progress.add_task(f"生成{step['name']}...", total=100)
                
                try:
                    df_processed = self._execute_label_step(df_processed, step)
                    
                    if show_progress:
                        progress.update(label_task, advance=100)
                        
                except Exception as e:
                    self.console.print(f"[red]错误: {step['name']}生成失败: {e}[/red]")
                    if show_progress:
                        progress.update(label_task, advance=100)
                    continue
            
            return df_processed
            
        finally:
            if show_progress:
                progress.stop()
    
    def _preprocess_polars(self, data: Union[pl.DataFrame, pl.LazyFrame], tickers: Optional[Set[str]]) -> pl.LazyFrame:
        """Polars数据预处理"""
        if isinstance(data, pl.DataFrame):
            df = data.lazy()
        else:
            df = data
        
        # 股票筛选
        if tickers:
            df = df.filter(pl.col("ticker").is_in(list(tickers)))
        
        # 确保时间列类型正确
        try:
            df = df.with_columns([
                pl.col("委托_datetime").cast(pl.Datetime("ns")),
                pl.col("事件_datetime").cast(pl.Datetime("ns"))
            ])
        except Exception as e:
            self.console.print(f"[yellow]警告: 时间列处理失败: {e}[/yellow]")
        
        return df
    
    def _preprocess_pandas(self, data: pd.DataFrame, tickers: Optional[Set[str]]) -> pd.DataFrame:
        """Pandas数据预处理"""
        df = data.copy()
        
        # 股票筛选
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        
        # 确保时间列类型正确
        try:
            df["委托_datetime"] = pd.to_datetime(df["委托_datetime"])
            df["事件_datetime"] = pd.to_datetime(df["事件_datetime"])
        except Exception as e:
            self.console.print(f"[yellow]警告: 时间列处理失败: {e}[/yellow]")
        
        return df
    
    def _execute_label_step(self, data: Union[pd.DataFrame, pl.LazyFrame], step: dict) -> Union[pd.DataFrame, pl.LazyFrame]:
        """执行单个标签生成步骤"""
        func_name = step["function"]
        
        try:
            if func_name == "apply_basic_spoofing_rules":
                if self.backend == "polars":
                    return apply_basic_spoofing_rules_polars(
                        data, self.r1_ms, self.r2_ms, self.r2_mult
                    )
                else:
                    return apply_basic_spoofing_rules_pandas(
                        data, self.r1_ms, self.r2_ms, self.r2_mult
                    )
                    
            elif func_name == "apply_enhanced_spoofing_rules":
                if self.backend == "pandas":
                    return apply_enhanced_spoofing_rules_pandas(
                        data, self.r1_ms, self.r2_ms, self.r2_mult
                    )
                else:
                    # 扩展规则目前只支持pandas，需要转换
                    if isinstance(data, pl.LazyFrame):
                        pd_data = data.collect().to_pandas()
                    else:
                        pd_data = data.to_pandas()
                    result = apply_enhanced_spoofing_rules_pandas(
                        pd_data, self.r1_ms, self.r2_ms, self.r2_mult
                    )
                    return pl.from_pandas(result)
            
            return data
            
        except Exception as e:
            self.console.print(f"[red]警告: {step['name']}生成失败: {e}[/red]")
            return data
    
    def process_single_file(self,
                           input_path: Path,
                           output_path: Path,
                           tickers: Optional[Set[str]] = None) -> Dict:
        """
        处理单个文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            tickers: 股票代码筛选
            
        Returns:
            处理结果统计
        """
        try:
            self.console.print(f"[cyan]📁 处理文件: {input_path.name}[/cyan]")
            
            # 加载数据
            if self.backend == "polars":
                df = pl.read_csv(input_path, try_parse_dates=True, infer_schema_length=10000, ignore_errors=True)
            else:
                df = pd.read_csv(input_path, parse_dates=["委托_datetime", "事件_datetime"], low_memory=False)
            
            # 检查数据是否为空
            if self.backend == "polars":
                if df.height == 0:
                    self.console.print(f"[yellow]跳过空文件: {input_path.name}[/yellow]")
                    return {"success": False, "reason": "empty_file"}
            else:
                if len(df) == 0:
                    self.console.print(f"[yellow]跳过空文件: {input_path.name}[/yellow]")
                    return {"success": False, "reason": "empty_file"}
            
            # 生成标签
            df_labels = self.generate_labels_for_data(df, tickers=tickers)
            
            # 使用明确定义的列选择（避免启发式判断出错）
            if self.backend == "polars":
                if isinstance(df_labels, pl.LazyFrame):
                    df_labels = df_labels.collect()
                
                # 获取实际存在的输出列
                available_cols = df_labels.columns
                final_cols = [col for col in get_label_output_columns(self.extended_rules) if col in available_cols]
                
                # 确保关键列存在
                missing_key_cols = [col for col in KEY_COLUMNS if col not in available_cols]
                if missing_key_cols:
                    self.console.print(f"[yellow]警告: 缺少关键列 {missing_key_cols}[/yellow]")
                
                df_final = df_labels.select(final_cols)
                
                # 保存结果
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_final.write_parquet(output_path)
                
                # 统计信息
                total_samples = df_final.height
                positive_samples = 0
                if "y_label" in df_final.columns:
                    positive_samples = df_final["y_label"].sum()
                elif any("label" in col for col in df_final.columns):
                    label_col = next(col for col in df_final.columns if "label" in col)
                    positive_samples = df_final[label_col].sum()
                
            else:
                # Pandas处理
                available_cols = df_labels.columns.tolist()
                final_cols = [col for col in get_label_output_columns(self.extended_rules) if col in available_cols]
                
                # 确保关键列存在
                missing_key_cols = [col for col in KEY_COLUMNS if col not in available_cols]
                if missing_key_cols:
                    self.console.print(f"[yellow]警告: 缺少关键列 {missing_key_cols}[/yellow]")
                
                df_final = df_labels[final_cols]
                
                # 保存结果
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_final.to_parquet(output_path, index=False)
                
                # 统计信息
                total_samples = len(df_final)
                positive_samples = 0
                if "y_label" in df_final.columns:
                    positive_samples = df_final["y_label"].sum()
                elif any("label" in col for col in df_final.columns):
                    label_col = next(col for col in df_final.columns if "label" in col)
                    positive_samples = df_final[label_col].sum()
            
            self.console.print(f"[green]✓ 已保存: {output_path.name}[/green]")
            self.console.print(f"  📊 样本数: {total_samples:,}, 正样本: {positive_samples}, 比例: {positive_samples/total_samples*100:.4f}%")
            
            return {
                "success": True,
                "total_samples": total_samples,
                "positive_samples": positive_samples,
                "positive_rate": positive_samples / total_samples if total_samples > 0 else 0
            }
            
        except Exception as e:
            self.console.print(f"[red]❌ 处理失败 {input_path.name}: {e}[/red]")
            return {"success": False, "reason": str(e)}
    
    def get_summary(self) -> dict:
        """获取标签生成器摘要信息"""
        return {
            "r1_ms": self.r1_ms,
            "r2_ms": self.r2_ms,
            "r2_mult": self.r2_mult,
            "extended_rules": self.extended_rules,
            "backend": self.backend,
            "pipeline_steps": len(self.label_pipeline),
            "pipeline_details": self.label_pipeline
        }

def process_event_stream_directory(event_stream_dir: Path,
                                  output_dir: Path,
                                  r1_ms: int = 50,
                                  r2_ms: int = 1000,
                                  r2_mult: float = 4.0,
                                  extended_rules: bool = False,
                                  backend: str = "polars",
                                  tickers: Optional[Set[str]] = None,
                                  dates: Optional[List[str]] = None) -> dict:
    """
    批量处理event_stream目录下的所有日期文件
    
    Args:
        event_stream_dir: event_stream根目录
        output_dir: 输出目录
        r1_ms: R1规则时间阈值
        r2_ms: R2规则时间阈值
        r2_mult: R2规则价差倍数
        extended_rules: 是否使用扩展标签规则
        backend: 计算后端
        tickers: 股票代码筛选
        dates: 指定处理日期
        
    Returns:
        处理结果统计
    """
    
    # 创建标签生成器
    generator = LabelGenerator(
        r1_ms=r1_ms, r2_ms=r2_ms, r2_mult=r2_mult,
        extended_rules=extended_rules, backend=backend
    )
    
    # 显示配置信息
    summary = generator.get_summary()
    console.print(f"\n[bold green]🏷️ 标签生成模块[/bold green]")
    console.print(f"[dim]输入目录: {event_stream_dir}[/dim]")
    console.print(f"[dim]输出目录: {output_dir}[/dim]")
    console.print(f"[dim]计算后端: {summary['backend']}[/dim]")
    console.print(f"[dim]R1阈值: {summary['r1_ms']}ms[/dim]")
    console.print(f"[dim]R2阈值: {summary['r2_ms']}ms (×{summary['r2_mult']})[/dim]")
    console.print(f"[dim]扩展规则: {summary['extended_rules']}[/dim]")
    console.print(f"[dim]股票筛选: {list(tickers) if tickers else '全部'}[/dim]")
    
    for step in summary['pipeline_details']:
        console.print(f"[dim]  • {step['name']}: {step['description']}[/dim]")
    
    # 查找所有委托事件流文件
    csv_files = list(event_stream_dir.glob("*/委托事件流.csv"))
    
    if not csv_files:
        console.print(f"[red]错误: 未找到委托事件流文件 {event_stream_dir}[/red]")
        return {"total": 0, "success": 0, "failed": 0}
    
    # 日期筛选
    if dates:
        target_dates = set(dates)
        csv_files = [f for f in csv_files if f.parent.name in target_dates]
        missing_dates = target_dates - {f.parent.name for f in csv_files}
        if missing_dates:
            console.print(f"[yellow]警告: 未找到日期 {missing_dates}[/yellow]")
    
    if not csv_files:
        console.print(f"[red]错误: 筛选后无可处理文件[/red]")
        return {"total": 0, "success": 0, "failed": 0}
    
    console.print(f"[dim]发现 {len(csv_files)} 个文件待处理[/dim]\n")
    
    # 批量处理
    results = {
        "total": len(csv_files), "success": 0, "failed": 0,
        "processed_files": [], "total_samples": 0, "total_positives": 0
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("处理文件", total=len(csv_files))
        
        for csv_file in sorted(csv_files):
            date_str = csv_file.parent.name
            output_file = output_dir / f"labels_{date_str}.parquet"
            
            progress.update(task, description=f"处理 {date_str}")
            
            result = generator.process_single_file(csv_file, output_file, tickers)
            
            if result["success"]:
                results["success"] += 1
                results["processed_files"].append(date_str)
                results["total_samples"] += result.get("total_samples", 0)
                results["total_positives"] += result.get("positive_samples", 0)
            else:
                results["failed"] += 1
            
            progress.advance(task)
    
    return results

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="标签生成模块")
    parser.add_argument("--input_dir", required=True, help="event_stream根目录")
    parser.add_argument("--output_dir", required=True, help="标签输出目录")
    parser.add_argument("--r1_ms", type=int, default=50, help="R1规则时间阈值(ms)")
    parser.add_argument("--r2_ms", type=int, default=1000, help="R2规则时间阈值(ms)")
    parser.add_argument("--r2_mult", type=float, default=4.0, help="R2规则价差倍数")
    parser.add_argument("--extended", action="store_true", help="使用扩展标签规则")
    parser.add_argument("--backend", choices=["pandas", "polars"], default="polars", help="计算后端")
    parser.add_argument("--tickers", nargs="*", help="股票代码筛选")
    parser.add_argument("--dates", nargs="*", help="指定处理日期，格式 YYYYMMDD")
    
    args = parser.parse_args()
    
    # 验证输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        console.print(f"[red]错误: 输入目录不存在 {input_dir}[/red]")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理参数
    tickers_set = set(args.tickers) if args.tickers else None
    
    try:
        # 执行批量处理
        results = process_event_stream_directory(
            event_stream_dir=input_dir,
            output_dir=output_dir,
            r1_ms=args.r1_ms,
            r2_ms=args.r2_ms,
            r2_mult=args.r2_mult,
            extended_rules=args.extended,
            backend=args.backend,
            tickers=tickers_set,
            dates=args.dates
        )
        
        # 显示结果
        console.print(f"\n[bold cyan]📊 处理完成[/bold cyan]")
        console.print(f"总文件数: {results['total']}")
        console.print(f"成功: {results['success']}")
        console.print(f"失败: {results['failed']}")
        
        if results['success'] > 0:
            console.print(f"[green]✅ 标签文件已保存到: {output_dir}[/green]")
            console.print(f"处理日期: {sorted(results['processed_files'])}")
            console.print(f"总样本数: {results['total_samples']:,}")
            console.print(f"正样本数: {results['total_positives']:,}")
            if results['total_samples'] > 0:
                positive_rate = results['total_positives'] / results['total_samples']
                console.print(f"整体正样本比例: {positive_rate*100:.4f}%")
        
        if results['failed'] > 0:
            console.print(f"[red]⚠️ {results['failed']} 个文件处理失败[/red]")
        
    except Exception as e:
        console.print(f"[red]❌ 处理出错: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
使用示例:

# 基础标签生成 (推荐使用polars后端)
python scripts/data_process/labels/label_generator.py \
    --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/labels" \
    --r1_ms 50 --r2_ms 1000 --r2_mult 4.0 \
    --backend polars

# 实际使用：扩展标签规则 (包含更多欺诈检测模式)
python scripts/data_process/labels/label_generator.py \
    --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/labels_enhanced" \
    --r1_ms 1000 --r2_ms 1000 --r2_mult 1.0 \
    --tickers 300233.SZ \
    --extended \
    --backend polars

# 指定股票和日期
python scripts/data_process/labels/label_generator.py \
    --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/labels" \
    --tickers 000001.SZ 000002.SZ \
    --dates 20240301 20240302 \
    --backend polars

# 批量处理所有数据
python scripts/data_process/labels/label_generator.py \
    --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/labels" \
    --backend polars
""" 