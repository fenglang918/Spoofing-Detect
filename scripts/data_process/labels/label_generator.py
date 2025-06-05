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
"""

import sys
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Set, List, Optional, Union, Dict
import argparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

# 导入标签生成组件
try:
    # 相对导入
    from ..labeling import improved_spoofing_rules_polars
    from ..enhanced_labeling_no_leakage import enhanced_spoofing_rules_no_leakage
    from ..utils import console
except ImportError:
    # 绝对导入
    sys.path.append(str(Path(__file__).parent.parent))
    from labeling import improved_spoofing_rules_polars
    from enhanced_labeling_no_leakage import enhanced_spoofing_rules_no_leakage
    from utils import console

console = Console()

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
                    "function": "enhanced_spoofing_rules_no_leakage",
                    "description": f"R1({self.r1_ms}ms) + R2({self.r2_ms}ms, {self.r2_mult}x) + 扩展规则"
                }
            ]
        else:
            pipeline = [
                {
                    "name": "基础标签规则", 
                    "function": "improved_spoofing_rules_polars",
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
            if func_name == "improved_spoofing_rules_polars":
                if self.backend == "polars":
                    return improved_spoofing_rules_polars(
                        data, self.r1_ms, self.r2_ms, self.r2_mult
                    )
                else:
                    # 如果是pandas，需要先转换
                    if isinstance(data, pd.DataFrame):
                        pl_data = pl.from_pandas(data).lazy()
                    else:
                        pl_data = data
                    result = improved_spoofing_rules_polars(
                        pl_data, self.r1_ms, self.r2_ms, self.r2_mult
                    )
                    return result.collect().to_pandas()
                    
            elif func_name == "enhanced_spoofing_rules_no_leakage":
                if self.backend == "pandas":
                    return enhanced_spoofing_rules_no_leakage(
                        data, self.r1_ms, self.r2_ms, self.r2_mult
                    )
                else:
                    # polars -> pandas -> polars
                    if isinstance(data, pl.LazyFrame):
                        pd_data = data.collect().to_pandas()
                    else:
                        pd_data = data.to_pandas()
                    result = enhanced_spoofing_rules_no_leakage(
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
                df = pl.read_csv(input_path, try_parse_dates=True)
            else:
                df = pd.read_csv(input_path, parse_dates=["委托_datetime", "事件_datetime"])
            
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
            
            # 提取标签列 (只保留主键 + 标签相关列)
            key_cols = ["自然日", "ticker", "交易所委托号"]
            
            if self.backend == "polars":
                if isinstance(df_labels, pl.LazyFrame):
                    df_labels = df_labels.collect()
                
                # 获取标签相关列
                label_cols = [col for col in df_labels.columns 
                             if any(keyword in col.lower() for keyword in 
                                   ['label', 'flag', 'spoofing', 'r1', 'r2']) and col not in key_cols]
                
                # 选择列
                final_cols = key_cols + label_cols
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
                label_cols = [col for col in df_labels.columns 
                             if any(keyword in col.lower() for keyword in 
                                   ['label', 'flag', 'spoofing', 'r1', 'r2']) and col not in key_cols]
                
                final_cols = key_cols + label_cols
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

# 基础标签生成
python scripts/data_process/labels/label_generator.py \
    --input_dir "/path/to/event_stream" \
    --output_dir "/path/to/labels" \
    --r1_ms 50 --r2_ms 1000 --r2_mult 4.0 \
    --backend polars

# 扩展标签规则
python scripts/data_process/labels/label_generator.py \
    --input_dir "/path/to/event_stream" \
    --output_dir "/path/to/labels" \
    --r1_ms 100 --r2_ms 1000 --r2_mult 1.5 \
    --extended \
    --backend pandas

# 指定股票和日期
python scripts/data_process/labels/label_generator.py \
    --input_dir "/path/to/event_stream" \
    --output_dir "/path/to/labels" \
    --tickers 000001.SZ 000002.SZ \
    --dates 20230301 20230302
""" 