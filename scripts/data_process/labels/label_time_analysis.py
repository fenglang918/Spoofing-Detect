#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标签时间分布分析
分析正样本在不同时间段的分布情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import time
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

console = Console()

def classify_time_period(time_obj):
    """分类时间段"""
    if pd.isna(time_obj):
        return "未知时间"
    
    if isinstance(time_obj, str):
        time_obj = pd.to_datetime(time_obj).time()
    elif hasattr(time_obj, 'time'):
        time_obj = time_obj.time()
    
    if time(9, 15) <= time_obj < time(9, 25):
        return "集合竞价早期"
    elif time(9, 25) <= time_obj < time(9, 30):
        return "集合竞价晚期"
    elif time(9, 30) <= time_obj <= time(10, 0):
        return "早盘开盘(9:30-10:00)"
    elif time(10, 0) < time_obj < time(11, 30):
        return "上午正常交易"
    elif time(11, 30) <= time_obj < time(13, 0):
        return "午休时间"
    elif time(13, 0) <= time_obj <= time(13, 15):
        return "午盘开盘(13:00-13:15)"
    elif time(13, 15) < time_obj < time(14, 45):
        return "下午正常交易"
    elif time(14, 45) <= time_obj <= time(15, 0):
        return "尾盘收盘"
    else:
        return "盘外时间"

def get_time_stats(time_obj):
    """获取时间统计信息"""
    if pd.isna(time_obj):
        return {"hour": None, "minute": None, "period": "未知时间"}
    
    if isinstance(time_obj, str):
        dt = pd.to_datetime(time_obj)
    else:
        dt = time_obj
    
    return {
        "hour": dt.hour,
        "minute": dt.minute,
        "period": classify_time_period(dt)
    }

def analyze_label_time_distribution(labels_dir: Path, output_dir: Path = None):
    """分析标签时间分布"""
    
    console.print(f"\n[bold green]📊 标签时间分布分析[/bold green]")
    console.print(f"[dim]标签目录: {labels_dir}[/dim]")
    
    # 查找所有标签文件
    label_files = list(labels_dir.glob("labels_*.parquet"))
    if not label_files:
        console.print(f"[red]❌ 未找到标签文件: {labels_dir}[/red]")
        return
    
    console.print(f"[dim]发现 {len(label_files)} 个标签文件[/dim]")
    
    # 合并所有数据
    all_data = []
    for file_path in sorted(label_files):
        try:
            df = pd.read_parquet(file_path)
            all_data.append(df)
        except Exception as e:
            console.print(f"[yellow]警告: 读取文件失败 {file_path.name}: {e}[/yellow]")
    
    if not all_data:
        console.print("[red]❌ 无有效数据文件[/red]")
        return
    
    # 合并数据
    df_all = pd.concat(all_data, ignore_index=True)
    console.print(f"[green]✅ 加载数据: {len(df_all):,} 条记录[/green]")
    
    # 数据预处理
    if '委托_datetime' in df_all.columns:
        df_all['委托_datetime'] = pd.to_datetime(df_all['委托_datetime'])
        df_all['委托_time'] = df_all['委托_datetime'].dt.time
        df_all['委托_hour'] = df_all['委托_datetime'].dt.hour
        df_all['委托_minute'] = df_all['委托_datetime'].dt.minute
        df_all['时间段'] = df_all['委托_time'].apply(classify_time_period)
    else:
        console.print("[red]❌ 缺少时间列[/red]")
        return
    
    # 识别标签列
    label_cols = [col for col in df_all.columns if 
                  any(keyword in col.lower() for keyword in ['label', 'flag', 'spoofing']) 
                  and col not in ['自然日', 'ticker']]
    
    if not label_cols:
        console.print("[red]❌ 未找到标签列[/red]")
        return
    
    console.print(f"[dim]标签列: {label_cols}[/dim]\n")
    
    # 1. 总体统计
    console.print("[bold cyan]📈 总体统计[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("指标")
    table.add_column("数值", justify="right")
    
    table.add_row("总样本数", f"{len(df_all):,}")
    table.add_row("时间范围", f"{df_all['委托_datetime'].min()} ~ {df_all['委托_datetime'].max()}")
    
    for col in label_cols:
        if col in df_all.columns:
            positive_count = df_all[col].sum() if df_all[col].dtype in ['int64', 'float64'] else 0
            positive_rate = positive_count / len(df_all) * 100
            table.add_row(f"{col} 正样本", f"{positive_count:,} ({positive_rate:.3f}%)")
    
    console.print(table)
    console.print()
    
    # 2. 时间段分布分析
    console.print("[bold cyan]⏰ 时间段分布分析[/bold cyan]")
    
    time_stats = df_all.groupby('时间段').agg({
        '委托_datetime': 'count',
        **{col: ['sum', 'mean'] for col in label_cols if col in df_all.columns}
    }).round(4)
    
    # 重命名列
    time_stats.columns = ['_'.join(col).strip('_') for col in time_stats.columns]
    time_stats = time_stats.rename(columns={'委托_datetime_count': '样本数'})
    
    console.print(time_stats)
    console.print()
    
    # 3. 小时级分布
    console.print("[bold cyan]🕐 小时级分布[/bold cyan]")
    hour_stats = df_all.groupby('委托_hour').agg({
        '委托_datetime': 'count',
        **{col: ['sum', 'mean'] for col in label_cols if col in df_all.columns}
    }).round(4)
    
    hour_stats.columns = ['_'.join(col).strip('_') for col in hour_stats.columns]
    hour_stats = hour_stats.rename(columns={'委托_datetime_count': '样本数'})
    
    console.print(hour_stats)
    console.print()
    
    # 4. 可视化分析
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold cyan]📊 生成可视化图表 -> {output_dir}[/bold cyan]")
        
        # 设置图形样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('标签时间分布分析', fontsize=16, fontweight='bold')
        
        # 子图1: 时间段样本数分布
        time_counts = df_all['时间段'].value_counts()
        axes[0, 0].bar(range(len(time_counts)), time_counts.values)
        axes[0, 0].set_xticks(range(len(time_counts)))
        axes[0, 0].set_xticklabels(time_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('各时间段样本数分布')
        axes[0, 0].set_ylabel('样本数')
        
        # 子图2: 主要标签的时间段分布
        main_label = 'y_label' if 'y_label' in label_cols else label_cols[0]
        if main_label in df_all.columns:
            positive_by_period = df_all[df_all[main_label] == 1]['时间段'].value_counts()
            axes[0, 1].bar(range(len(positive_by_period)), positive_by_period.values, color='red', alpha=0.7)
            axes[0, 1].set_xticks(range(len(positive_by_period)))
            axes[0, 1].set_xticklabels(positive_by_period.index, rotation=45, ha='right')
            axes[0, 1].set_title(f'{main_label} 正样本时间段分布')
            axes[0, 1].set_ylabel('正样本数')
        
        # 子图3: 小时级分布
        hour_counts = df_all['委托_hour'].value_counts().sort_index()
        axes[1, 0].plot(hour_counts.index, hour_counts.values, marker='o')
        axes[1, 0].set_title('小时级样本分布')
        axes[1, 0].set_xlabel('小时')
        axes[1, 0].set_ylabel('样本数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 正样本率时间分布
        if main_label in df_all.columns:
            hour_positive_rate = df_all.groupby('委托_hour')[main_label].mean() * 100
            axes[1, 1].bar(hour_positive_rate.index, hour_positive_rate.values, alpha=0.7, color='orange')
            axes[1, 1].set_title(f'{main_label} 各小时正样本率')
            axes[1, 1].set_xlabel('小时')
            axes[1, 1].set_ylabel('正样本率 (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / "label_time_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ 图表已保存: {plot_file}[/green]")
        
        # 保存详细统计数据
        stats_file = output_dir / "time_distribution_stats.xlsx"
        with pd.ExcelWriter(stats_file) as writer:
            time_stats.to_excel(writer, sheet_name='时间段统计')
            hour_stats.to_excel(writer, sheet_name='小时统计')
            
            # 添加原始数据样本
            sample_data = df_all[['自然日', 'ticker', '委托_datetime', '时间段', '委托数量'] + label_cols].head(1000)
            sample_data.to_excel(writer, sheet_name='数据样本', index=False)
        
        console.print(f"[green]✅ 统计数据已保存: {stats_file}[/green]")
    
    # 5. 关键发现总结
    console.print("[bold yellow]🔍 关键发现[/bold yellow]")
    
    main_label = 'y_label' if 'y_label' in label_cols else label_cols[0]
    if main_label in df_all.columns:
        # 检查开盘时间段正样本比例
        opening_periods = ['早盘开盘(9:30-10:00)', '午盘开盘(13:00-13:15)']
        opening_data = df_all[df_all['时间段'].isin(opening_periods)]
        
        if len(opening_data) > 0:
            opening_positive = opening_data[main_label].sum()
            opening_total = len(opening_data)
            opening_rate = opening_positive / opening_total * 100
            
            console.print(f"• 开盘时间段样本: {opening_total:,} 条")
            console.print(f"• 开盘时间段正样本: {opening_positive} 条 ({opening_rate:.3f}%)")
            
            if opening_rate < 1.0:
                console.print("[green]✅ 开盘时间段正样本比例较低，修复有效[/green]")
            else:
                console.print("[yellow]⚠️ 开盘时间段正样本比例仍然较高[/yellow]")
        
        # 检查异常时间段
        abnormal_periods = ['集合竞价晚期', '午休时间', '盘外时间']
        abnormal_data = df_all[df_all['时间段'].isin(abnormal_periods)]
        
        if len(abnormal_data) > 0:
            abnormal_positive = abnormal_data[main_label].sum()
            abnormal_total = len(abnormal_data)
            abnormal_rate = abnormal_positive / abnormal_total * 100
            
            console.print(f"• 异常时间段样本: {abnormal_total:,} 条")
            console.print(f"• 异常时间段正样本: {abnormal_positive} 条 ({abnormal_rate:.3f}%)")
    
    return df_all

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="标签时间分布分析")
    parser.add_argument("--labels_dir", required=True, help="标签文件目录")
    parser.add_argument("--output_dir", help="输出目录(可选)")
    
    args = parser.parse_args()
    
    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        console.print(f"[red]❌ 标签目录不存在: {labels_dir}[/red]")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # 执行分析
    analyze_label_time_distribution(labels_dir, output_dir)

if __name__ == "__main__":
    main() 


"""
python scripts/data_process/labels/label_time_analysis.py \
    --labels_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/labels_enhanced" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/labels_time_analysis"
"""