#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_generator.py
────────────────────────────────────────
特征生成模块 - 第二阶段特征计算
• 从委托事件流计算各类特征
• 支持批量处理多日期数据
• 模块化特征计算流水线
• 保留所有计算特征（不过滤）
"""

import sys
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Set, List, Optional, Union
import argparse
import glob
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

# 导入特征计算组件
try:
    # 相对导入
    from ..feature_engineering import (
        calc_realtime_features, calculate_enhanced_realtime_features,
        calculate_order_book_pressure, calc_realtime_features_polars,
        calculate_order_book_pressure_polars
    )
    from ..utils import console
except ImportError:
    # 绝对导入
    sys.path.append(str(Path(__file__).parent.parent))
    from feature_engineering import (
        calc_realtime_features, calculate_enhanced_realtime_features,
        calculate_order_book_pressure, calc_realtime_features_polars,
        calculate_order_book_pressure_polars
    )
    from utils import console

console = Console()

class FeatureGenerator:
    """特征生成器"""
    
    def __init__(self, backend: str = "polars", extended_features: bool = True):
        """
        初始化特征生成器
        
        Args:
            backend: 计算后端 ("polars" 或 "pandas")
            extended_features: 是否计算扩展特征
        """
        self.backend = backend
        self.extended_features = extended_features
        self.console = Console()
        
        # 定义特征计算流水线
        self.feature_pipeline = self._build_feature_pipeline()
    
    def _build_feature_pipeline(self) -> List[dict]:
        """构建特征计算流水线"""
        if self.backend == "polars":
            pipeline = [
                {
                    "name": "基础实时特征",
                    "function": "calc_realtime_features_polars",
                    "description": "盘口快照、价格特征、滚动窗口统计"
                },
                {
                    "name": "订单簿压力特征", 
                    "function": "calculate_order_book_pressure_polars",
                    "description": "book_imbalance、price_aggressiveness等"
                }
            ]
        else:
            pipeline = [
                {
                    "name": "基础实时特征",
                    "function": "calc_realtime_features",
                    "description": "盘口快照、价格特征、滚动窗口统计"
                }
            ]
            
            if self.extended_features:
                pipeline.append({
                    "name": "扩展实时特征",
                    "function": "calculate_enhanced_realtime_features", 
                    "description": "z_survival、价格动量、订单密度等"
                })
            
            pipeline.append({
                "name": "订单簿压力特征",
                "function": "calculate_order_book_pressure",
                "description": "book_imbalance、price_aggressiveness等"
            })
        
        return pipeline
    
    def generate_features_for_data(self, 
                                  data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
                                  tickers: Optional[Set[str]] = None,
                                  show_progress: bool = True) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        为给定数据生成特征
        
        Args:
            data: 输入的委托事件流数据
            tickers: 股票代码筛选
            show_progress: 是否显示进度条
            
        Returns:
            带特征的DataFrame
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
            
            # 特征计算流水线
            for i, step in enumerate(self.feature_pipeline):
                if show_progress:
                    feat_task = progress.add_task(f"计算{step['name']}...", total=100)
                
                try:
                    df_processed = self._execute_feature_step(df_processed, step)
                    
                    if show_progress:
                        progress.update(feat_task, advance=100)
                        
                except Exception as e:
                    self.console.print(f"[red]错误: {step['name']}计算失败: {e}[/red]")
                    if show_progress:
                        progress.update(feat_task, advance=100)
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
            # 安全的时间转换
            df = df.with_columns([
                pl.col("委托_datetime").str.strptime(pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f", strict=False),
                pl.col("事件_datetime").str.strptime(pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f", strict=False)
            ])
            
            # 按委托时间排序（重要：解决rolling操作的排序问题）
            df = df.sort("委托_datetime")
            
        except Exception as e:
            self.console.print(f"[yellow]警告: 时间列处理失败: {e}[/yellow]")
            # 如果时间转换失败，尝试其他格式
            try:
                df = df.with_columns([
                    pl.col("委托_datetime").cast(pl.Datetime("ns")),
                    pl.col("事件_datetime").cast(pl.Datetime("ns"))
                ])
                df = df.sort("委托_datetime")
            except Exception as e2:
                self.console.print(f"[red]错误: 无法转换时间列: {e2}[/red]")
        
        return df
    
    def _preprocess_pandas(self, data: pd.DataFrame, tickers: Optional[Set[str]]) -> pd.DataFrame:
        """Pandas数据预处理"""
        df = data.copy()
        
        # 股票筛选
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        
        # 确保时间列类型正确
        try:
            df["委托_datetime"] = pd.to_datetime(df["委托_datetime"], errors='coerce')
            df["事件_datetime"] = pd.to_datetime(df["事件_datetime"], errors='coerce')
            
            # 删除时间转换失败的行
            df = df.dropna(subset=["委托_datetime", "事件_datetime"])
            
            # 按委托时间排序
            df = df.sort_values("委托_datetime")
            
        except Exception as e:
            self.console.print(f"[yellow]警告: 时间列处理失败: {e}[/yellow]")
        
        return df
    
    def _execute_feature_step(self, data: Union[pd.DataFrame, pl.LazyFrame], step: dict) -> Union[pd.DataFrame, pl.LazyFrame]:
        """执行单个特征计算步骤"""
        func_name = step["function"]
        
        try:
            if self.backend == "polars":
                if func_name == "calc_realtime_features_polars":
                    return calc_realtime_features_polars(data)
                elif func_name == "calculate_order_book_pressure_polars":
                    # 获取当前schema
                    if isinstance(data, pl.LazyFrame):
                        schema = data.collect_schema()
                    else:
                        schema = data.schema
                    return calculate_order_book_pressure_polars(data, dict(schema))
            else:
                if func_name == "calc_realtime_features":
                    return calc_realtime_features(data)
                elif func_name == "calculate_enhanced_realtime_features":
                    return calculate_enhanced_realtime_features(data)
                elif func_name == "calculate_order_book_pressure":
                    return calculate_order_book_pressure(data)
            
            return data
            
        except Exception as e:
            self.console.print(f"[red]警告: {step['name']}计算失败: {e}[/red]")
            return data
    
    def process_single_file(self, 
                           input_path: Path,
                           output_path: Path,
                           tickers: Optional[Set[str]] = None) -> bool:
        """
        处理单个文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            tickers: 股票代码筛选
            
        Returns:
            是否处理成功
        """
        try:
            self.console.print(f"[cyan]📁 处理文件: {input_path.name}[/cyan]")
            
            # 加载数据
            if self.backend == "polars":
                # 定义schema_overrides来处理问题列
                schema_overrides = {
                    "委托类型": pl.Utf8,  # 强制为字符串类型
                    "事件类型": pl.Utf8,
                    "方向_委托": pl.Utf8,
                    "方向_事件": pl.Utf8,
                    "成交代码": pl.Utf8,
                    "委托_datetime": pl.Utf8,  # 先读为字符串，后续转换
                    "事件_datetime": pl.Utf8
                }
                
                df = pl.read_csv(
                    input_path, 
                    try_parse_dates=False,  # 关闭自动日期解析
                    schema_overrides=schema_overrides,
                    infer_schema_length=10000,  # 增加类型推断长度
                    ignore_errors=True  # 忽略解析错误
                )
            else:
                df = pd.read_csv(input_path, dtype=str)  # 先全部读为字符串
            
            # 检查数据是否为空
            if self.backend == "polars":
                if df.height == 0:
                    self.console.print(f"[yellow]跳过空文件: {input_path.name}[/yellow]")
                    return False
            else:
                if len(df) == 0:
                    self.console.print(f"[yellow]跳过空文件: {input_path.name}[/yellow]")
                    return False
            
            # 生成特征
            df_features = self.generate_features_for_data(df, tickers=tickers)
            
            # 保存结果
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.backend == "polars":
                if isinstance(df_features, pl.LazyFrame):
                    df_features = df_features.collect()
                df_features.write_parquet(output_path)
            else:
                df_features.to_parquet(output_path, index=False)
            
            # 显示统计信息
            if self.backend == "polars":
                feature_count = len([c for c in df_features.columns if c not in ['自然日', 'ticker', '交易所委托号']])
                sample_count = df_features.height
            else:
                feature_count = len([c for c in df_features.columns if c not in ['自然日', 'ticker', '交易所委托号']])
                sample_count = len(df_features)
            
            self.console.print(f"[green]✓ 已保存: {output_path.name}[/green]")
            self.console.print(f"  📊 特征数: {feature_count}, 样本数: {sample_count:,}")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ 处理失败 {input_path.name}: {e}[/red]")
            return False
    
    def get_summary(self) -> dict:
        """获取特征生成器摘要信息"""
        return {
            "backend": self.backend,
            "extended_features": self.extended_features,
            "pipeline_steps": len(self.feature_pipeline),
            "pipeline_details": self.feature_pipeline
        }

def process_event_stream_directory(event_stream_dir: Path,
                                  output_dir: Path,
                                  backend: str = "polars",
                                  extended_features: bool = True,
                                  tickers: Optional[Set[str]] = None,
                                  dates: Optional[List[str]] = None) -> dict:
    """
    批量处理event_stream目录下的所有日期文件
    
    Args:
        event_stream_dir: event_stream根目录
        output_dir: 输出目录
        backend: 计算后端
        extended_features: 是否计算扩展特征
        tickers: 股票代码筛选
        dates: 指定处理日期
        
    Returns:
        处理结果统计
    """
    
    # 创建特征生成器
    generator = FeatureGenerator(backend=backend, extended_features=extended_features)
    
    # 显示配置信息
    summary = generator.get_summary()
    console.print(f"\n[bold green]🎯 特征生成模块[/bold green]")
    console.print(f"[dim]输入目录: {event_stream_dir}[/dim]")
    console.print(f"[dim]输出目录: {output_dir}[/dim]")
    console.print(f"[dim]计算后端: {summary['backend']}[/dim]")
    console.print(f"[dim]扩展特征: {summary['extended_features']}[/dim]")
    console.print(f"[dim]流水线步骤: {summary['pipeline_steps']}[/dim]")
    console.print(f"[dim]股票筛选: {list(tickers) if tickers else '全部'}[/dim]")
    
    for step in summary['pipeline_details']:
        console.print(f"[dim]  • {step['name']}: {step['description']}[/dim]")
    
    # 查找所有委托事件流文件
    pattern = event_stream_dir / "*" / "委托事件流.csv"
    csv_files = list(event_stream_dir.glob("*/委托事件流.csv"))
    
    if not csv_files:
        console.print(f"[red]错误: 未找到委托事件流文件 {pattern}[/red]")
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
    results = {"total": len(csv_files), "success": 0, "failed": 0, "processed_files": []}
    
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
            output_file = output_dir / f"X_{date_str}.parquet"
            
            progress.update(task, description=f"处理 {date_str}")
            
            if generator.process_single_file(csv_file, output_file, tickers):
                results["success"] += 1
                results["processed_files"].append(date_str)
            else:
                results["failed"] += 1
            
            progress.advance(task)
    
    return results

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="特征生成模块")
    parser.add_argument("--input_dir", required=True, help="event_stream根目录")
    parser.add_argument("--output_dir", required=True, help="特征输出目录")
    parser.add_argument("--backend", choices=["pandas", "polars"], default="polars", help="计算后端")
    parser.add_argument("--extended", action="store_true", help="计算扩展特征")
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
            backend=args.backend,
            extended_features=args.extended,
            tickers=tickers_set,
            dates=args.dates
        )
        
        # 显示结果
        console.print(f"\n[bold cyan]📊 处理完成[/bold cyan]")
        console.print(f"总文件数: {results['total']}")
        console.print(f"成功: {results['success']}")
        console.print(f"失败: {results['failed']}")
        
        if results['success'] > 0:
            console.print(f"[green]✅ 特征文件已保存到: {output_dir}[/green]")
            console.print(f"处理日期: {sorted(results['processed_files'])}")
        
        if results['failed'] > 0:
            console.print(f"[red]⚠️ {results['failed']} 个文件处理失败[/red]")
        
    except Exception as e:
        console.print(f"[red]❌ 处理出错: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
使用示例:

# 处理所有日期文件
python scripts/data_process/features/feature_generator.py \
    --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/features" \
    --backend polars \
    --extended

# 处理指定股票和日期
python scripts/data_process/features/feature_generator.py \
    --input_dir "/path/to/event_stream" \
    --output_dir "/path/to/features" \
    --tickers 000001.SZ 000002.SZ \
    --dates 20230301 20230302 \
    --backend polars
""" 