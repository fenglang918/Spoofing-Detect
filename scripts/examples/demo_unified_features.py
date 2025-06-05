#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_unified_features.py
──────────────────────────────────────────────────
统一特征计算模块使用示例
演示如何在不同场景下使用 UnifiedFeatureCalculator
"""

import sys
from pathlib import Path
import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from data_process.unified_features import UnifiedFeatureCalculator

console = Console()

def demo_basic_usage():
    """演示基础用法"""
    console.print("\n[bold green]📋 Demo 1: 基础用法[/bold green]")
    
    # 创建特征计算器
    calculator = UnifiedFeatureCalculator(backend="polars", extended_features=True)
    
    # 显示配置信息
    summary = calculator.get_feature_summary()
    console.print(f"配置: {summary}")

def demo_pandas_backend():
    """演示Pandas后端"""
    console.print("\n[bold blue]📋 Demo 2: Pandas后端[/bold blue]")
    
    # 创建模拟数据
    sample_data = pd.DataFrame({
        'ticker': ['000001.SZ'] * 100,
        '委托_datetime': pd.date_range('2024-01-01 09:30:00', periods=100, freq='1s'),
        '事件_datetime': pd.date_range('2024-01-01 09:30:00', periods=100, freq='1s'),
        '委托价格': [10.0 + i * 0.01 for i in range(100)],
        '委托数量': [1000] * 100,
        '方向_委托': ['买', '卖'] * 50,
        '事件类型': ['委托'] * 100,
        '申买价1': [9.98] * 100,
        '申卖价1': [10.02] * 100,
        '前收盘': [10.0] * 100,
        '申买量1': [5000] * 100,
        '申卖量1': [4000] * 100,
        '交易所委托号': [f'order_{i}' for i in range(100)],
        '存活时间_ms': [100] * 100,
    })
    
    # 使用Pandas后端计算特征
    calculator = UnifiedFeatureCalculator(backend="pandas", extended_features=True)
    result = calculator.calculate_features(sample_data, show_progress=True)
    
    console.print(f"[green]✅ Pandas结果: {result.shape}[/green]")
    console.print(f"特征列: {list(result.columns)[:10]}...")

def demo_polars_backend():
    """演示Polars后端"""
    console.print("\n[bold cyan]📋 Demo 3: Polars后端[/bold cyan]")
    
    # 创建Polars模拟数据
    sample_data = pl.DataFrame({
        'ticker': ['000001.SZ'] * 100,
        '委托_datetime': pl.date_range(pl.datetime(2024, 1, 1, 9, 30), pl.datetime(2024, 1, 1, 9, 31, 39), "1s", eager=True),
        '事件_datetime': pl.date_range(pl.datetime(2024, 1, 1, 9, 30), pl.datetime(2024, 1, 1, 9, 31, 39), "1s", eager=True),
        '委托价格': [10.0 + i * 0.01 for i in range(100)],
        '委托数量': [1000] * 100,
        '方向_委托': ['买', '卖'] * 50,
        '事件类型': ['委托'] * 100,
        '申买价1': [9.98] * 100,
        '申卖价1': [10.02] * 100,
        '前收盘': [10.0] * 100,
        '申买量1': [5000] * 100,
        '申卖量1': [4000] * 100,
        '交易所委托号': [f'order_{i}' for i in range(100)],
        '存活时间_ms': [100] * 100,
    })
    
    # 使用Polars后端计算特征
    calculator = UnifiedFeatureCalculator(backend="polars", extended_features=True)
    result = calculator.calculate_features(sample_data, show_progress=True)
    
    console.print(f"[green]✅ Polars结果: {result.shape}[/green]")
    console.print(f"特征列: {list(result.columns)[:10]}...")

def demo_ticker_filtering():
    """演示股票筛选功能"""
    console.print("\n[bold magenta]📋 Demo 4: 股票筛选[/bold magenta]")
    
    # 创建多股票数据
    tickers = ['000001.SZ', '000002.SZ', '000003.SZ']
    sample_data = []
    
    for ticker in tickers:
        for i in range(50):
            sample_data.append({
                'ticker': ticker,
                '委托_datetime': pd.Timestamp('2024-01-01 09:30:00') + pd.Timedelta(seconds=i),
                '事件_datetime': pd.Timestamp('2024-01-01 09:30:00') + pd.Timedelta(seconds=i),
                '委托价格': 10.0 + i * 0.01,
                '委托数量': 1000,
                '方向_委托': '买' if i % 2 == 0 else '卖',
                '事件类型': '委托',
                '申买价1': 9.98,
                '申卖价1': 10.02,
                '前收盘': 10.0,
                '申买量1': 5000,
                '申卖量1': 4000,
                '交易所委托号': f'{ticker}_order_{i}',
                '存活时间_ms': 100,
            })
    
    df = pd.DataFrame(sample_data)
    
    # 只计算特定股票的特征
    calculator = UnifiedFeatureCalculator(backend="pandas", extended_features=False)
    result = calculator.calculate_features(df, tickers={'000001.SZ', '000002.SZ'}, show_progress=True)
    
    console.print(f"[green]✅ 筛选后结果: {result.shape}[/green]")
    console.print(f"包含股票: {result['ticker'].unique() if 'ticker' in result.columns else 'No ticker column'}")

def demo_feature_comparison():
    """演示特征对比"""
    console.print("\n[bold yellow]📋 Demo 5: 特征对比[/bold yellow]")
    
    # 创建样本数据
    sample_data = pd.DataFrame({
        'ticker': ['000001.SZ'] * 10,
        '委托_datetime': pd.date_range('2024-01-01 09:30:00', periods=10, freq='1s'),
        '事件_datetime': pd.date_range('2024-01-01 09:30:00', periods=10, freq='1s'),
        '委托价格': [10.0 + i * 0.01 for i in range(10)],
        '委托数量': [1000] * 10,
        '方向_委托': ['买', '卖'] * 5,
        '事件类型': ['委托'] * 10,
        '申买价1': [9.98] * 10,
        '申卖价1': [10.02] * 10,
        '前收盘': [10.0] * 10,
        '申买量1': [5000] * 10,
        '申卖量1': [4000] * 10,
        '交易所委托号': [f'order_{i}' for i in range(10)],
        '存活时间_ms': [100] * 10,
    })
    
    # 基础特征 vs 扩展特征
    calc_basic = UnifiedFeatureCalculator(backend="pandas", extended_features=False)
    calc_extended = UnifiedFeatureCalculator(backend="pandas", extended_features=True)
    
    result_basic = calc_basic.calculate_features(sample_data, show_progress=False)
    result_extended = calc_extended.calculate_features(sample_data, show_progress=False)
    
    table = Table(title="特征对比")
    table.add_column("模式", style="cyan")
    table.add_column("特征数量", style="magenta")
    table.add_column("数据形状", style="green")
    
    table.add_row("基础特征", str(len(result_basic.columns)), str(result_basic.shape))
    table.add_row("扩展特征", str(len(result_extended.columns)), str(result_extended.shape))
    
    console.print(table)

def demo_csv_processing():
    """演示CSV文件处理"""
    console.print("\n[bold red]📋 Demo 6: CSV文件处理模拟[/bold red]")
    
    # 创建临时CSV文件
    temp_csv = Path("/tmp/demo_events.csv")
    
    sample_data = pd.DataFrame({
        'ticker': ['000001.SZ'] * 20,
        '委托_datetime': pd.date_range('2024-01-01 09:30:00', periods=20, freq='10s'),
        '事件_datetime': pd.date_range('2024-01-01 09:30:00', periods=20, freq='10s'),
        '委托价格': [10.0 + i * 0.01 for i in range(20)],
        '委托数量': [1000] * 20,
        '方向_委托': ['买', '卖'] * 10,
        '事件类型': ['委托'] * 20,
        '申买价1': [9.98] * 20,
        '申卖价1': [10.02] * 20,
        '前收盘': [10.0] * 20,
        '申买量1': [5000] * 20,
        '申卖量1': [4000] * 20,
        '交易所委托号': [f'order_{i}' for i in range(20)],
        '存活时间_ms': [100] * 20,
    })
    
    sample_data.to_csv(temp_csv, index=False)
    
    # 使用CSV处理功能
    calculator = UnifiedFeatureCalculator(backend="polars", extended_features=True)
    
    console.print(f"[yellow]处理CSV文件: {temp_csv}[/yellow]")
    result = calculator.process_csv_file(
        csv_path=temp_csv,
        tickers={'000001.SZ'},
        output_path="/tmp/demo_features.parquet"
    )
    
    # 清理临时文件
    temp_csv.unlink(missing_ok=True)
    Path("/tmp/demo_features.parquet").unlink(missing_ok=True)

def main():
    """运行所有演示"""
    console.print("[bold green]🚀 统一特征计算模块演示[/bold green]")
    console.print("=" * 60)
    
    try:
        demo_basic_usage()
        demo_pandas_backend()
        demo_polars_backend()
        demo_ticker_filtering()
        demo_feature_comparison()
        demo_csv_processing()
        
        console.print("\n[bold green]✅ 所有演示完成！[/bold green]")
        console.print("\n[cyan]💡 使用建议:[/cyan]")
        console.print("1. 对于大数据量，推荐使用Polars后端")
        console.print("2. 扩展特征提供更丰富的信息，但计算时间更长")
        console.print("3. 股票筛选可以显著减少计算量")
        console.print("4. 特征白名单确保生产环境的一致性")
        
    except Exception as e:
        console.print(f"[red]❌ 演示过程中出现错误: {e}[/red]")

if __name__ == "__main__":
    main()

"""
运行示例:
python scripts/examples/demo_unified_features.py
""" 