#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
complete_workflow.py
─────────────────────────────────────────────────
完整工作流程示例 - 从ETL到特征分析的端到端处理
• 分日期特征计算（不过滤）
• 整合所有日期数据
• 全局特征质量分析
• 智能特征过滤
• 模型就绪数据生成
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import argparse

console = Console()

def run_etl_pipeline(data_root: str, tickers: list = None, backend: str = "polars"):
    """运行ETL流水线"""
    console.print(f"\n[bold green]🚀 Step 1: 运行ETL流水线[/bold green]")
    
    cmd = [
        "python", "scripts/data_process/run_etl_from_event_refactored.py",
        "--root", f"{data_root}/event_stream",
        "--backend", backend,
        "--extended_labels"
    ]
    
    if tickers:
        cmd.extend(["--tickers"] + tickers)
    
    console.print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        console.print("[green]✅ ETL流水线执行成功[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ ETL流水线执行失败: {e}[/red]")
        console.print(f"错误输出: {e.stderr}")
        return False

def run_feature_analysis(data_root: str, save_filtered: bool = True):
    """运行特征质量分析"""
    console.print(f"\n[bold green]🔍 Step 2: 特征质量分析[/bold green]")
    
    analysis_dir = Path(data_root) / "analysis_results"
    analysis_dir.mkdir(exist_ok=True)
    
    cmd = [
        "python", "scripts/data_process/feature_analysis.py",
        "--data_root", data_root,
        "--target_col", "y_label",
        "--output_dir", str(analysis_dir)
    ]
    
    if save_filtered:
        cmd.append("--save_filtered")
    
    console.print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        console.print("[green]✅ 特征分析完成[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ 特征分析失败: {e}[/red]")
        console.print(f"错误输出: {e.stderr}")
        return False

def generate_training_data(data_root: str):
    """生成训练就绪的数据"""
    console.print(f"\n[bold green]📊 Step 3: 生成训练数据[/bold green]")
    
    # 这里可以添加额外的数据处理步骤
    # 比如：
    # - 特征标准化
    # - 数据分割（训练/验证/测试）
    # - 样本平衡处理等
    
    filtered_data_path = Path(data_root) / "filtered_data.parquet"
    if filtered_data_path.exists():
        console.print(f"[green]✅ 训练数据已准备: {filtered_data_path}[/green]")
        
        # 显示数据基本信息
        import pandas as pd
        df = pd.read_parquet(filtered_data_path)
        console.print(f"  数据形状: {df.shape}")
        console.print(f"  特征数量: {len([c for c in df.columns if c not in ['自然日', 'ticker', '交易所委托号', 'y_label']])}")
        console.print(f"  样本数量: {len(df):,}")
        
        if 'y_label' in df.columns:
            positive_rate = df['y_label'].mean()
            console.print(f"  正样本比例: {positive_rate:.4f}")
        
        return True
    else:
        console.print(f"[red]❌ 未找到过滤后的数据文件[/red]")
        return False

def create_data_summary(data_root: str):
    """创建数据摘要报告"""
    console.print(f"\n[bold green]📋 Step 4: 生成数据摘要[/bold green]")
    
    analysis_dir = Path(data_root) / "analysis_results"
    summary_file = analysis_dir / "data_summary.md"
    
    try:
        import pandas as pd
        import json
        
        # 读取分析结果
        with open(analysis_dir / "feature_analysis.json", "r", encoding="utf-8") as f:
            analysis_results = json.load(f)
        
        # 读取过滤后的数据
        filtered_data = pd.read_parquet(Path(data_root) / "filtered_data.parquet")
        
        # 生成摘要报告
        summary_content = f"""# 数据处理摘要报告

## 📊 基础统计

- **数据根目录**: {data_root}
- **最终数据形状**: {filtered_data.shape}
- **特征数量**: {len([c for c in filtered_data.columns if c not in ['自然日', 'ticker', '交易所委托号', 'y_label']])}
- **样本数量**: {len(filtered_data):,}
- **内存使用**: {filtered_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB

## 🚫 移除的特征

### 问题特征统计
- **常数列**: {len(analysis_results.get('problematic_features', {}).get('constant_features', []))} 个
- **低方差特征**: {len(analysis_results.get('problematic_features', {}).get('low_variance_features', []))} 个  
- **高缺失率特征**: {len(analysis_results.get('problematic_features', {}).get('high_missing_features', []))} 个
- **重复特征**: {len(analysis_results.get('problematic_features', {}).get('duplicate_features', []))} 个

### 信息泄露特征
- **疑似泄露特征**: {len(analysis_results.get('leakage_features', {}).get('suspicious_features', []))} 个
- **模式匹配特征**: {sum(len(matches) for matches in analysis_results.get('leakage_features', {}).get('pattern_matches', {}).values())} 个

## ✅ 建议策略

- **建议移除特征数**: {len(analysis_results.get('recommendations', {}).get('features_to_remove', []))} 个
- **需调查特征数**: {len(analysis_results.get('recommendations', {}).get('features_to_investigate', []))} 个
- **优先特征数**: {len(analysis_results.get('recommendations', {}).get('priority_features', []))} 个
- **安全核心特征数**: {len(analysis_results.get('recommendations', {}).get('safe_features', []))} 个

## 🎯 标签分布

"""
        
        if 'y_label' in filtered_data.columns:
            positive_count = filtered_data['y_label'].sum()
            total_count = len(filtered_data)
            positive_rate = positive_count / total_count
            
            summary_content += f"""- **正样本数量**: {positive_count:,}
- **负样本数量**: {total_count - positive_count:,}
- **正样本比例**: {positive_rate:.4f}
"""
        
        summary_content += f"""
## 📈 下一步建议

1. **特征工程**: 考虑基于核心特征构造新的衍生特征
2. **模型选择**: 根据特征类型选择合适的模型算法  
3. **样本平衡**: 根据标签分布考虑采样策略
4. **交叉验证**: 按时间/股票进行分层交叉验证
5. **特征选择**: 可进一步使用模型内置的特征选择方法

## 📁 生成的文件

- `filtered_data.parquet`: 过滤后的训练数据
- `analysis_results/feature_analysis.json`: 详细分析结果
- `analysis_results/data_summary.md`: 本摘要报告

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_content)
        
        console.print(f"[green]📄 摘要报告已生成: {summary_file}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ 摘要生成失败: {e}[/red]")
        return False

def main():
    """完整工作流程主函数"""
    parser = argparse.ArgumentParser(description="完整数据处理工作流程")
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--tickers", nargs="*", help="股票代码筛选")
    parser.add_argument("--backend", choices=["pandas", "polars"], default="polars", help="ETL后端")
    parser.add_argument("--skip_etl", action="store_true", help="跳过ETL步骤")
    parser.add_argument("--skip_analysis", action="store_true", help="跳过特征分析步骤")
    
    args = parser.parse_args()
    
    console.print(f"[bold cyan]🎯 开始完整数据处理工作流程[/bold cyan]")
    console.print(f"数据根目录: {args.data_root}")
    console.print(f"股票筛选: {args.tickers if args.tickers else '全部'}")
    console.print(f"处理后端: {args.backend}")
    
    success_steps = 0
    total_steps = 4
    
    # Step 1: ETL Pipeline
    if not args.skip_etl:
        if run_etl_pipeline(args.data_root, args.tickers, args.backend):
            success_steps += 1
    else:
        console.print(f"[yellow]⏭️  跳过ETL步骤[/yellow]")
        success_steps += 1
    
    # Step 2: Feature Analysis
    if not args.skip_analysis:
        if run_feature_analysis(args.data_root, save_filtered=True):
            success_steps += 1
    else:
        console.print(f"[yellow]⏭️  跳过特征分析步骤[/yellow]")
        success_steps += 1
    
    # Step 3: Generate Training Data
    if generate_training_data(args.data_root):
        success_steps += 1
    
    # Step 4: Create Summary
    if create_data_summary(args.data_root):
        success_steps += 1
    
    # 总结
    console.print(f"\n[bold green]🎉 工作流程完成[/bold green]")
    console.print(f"成功步骤: {success_steps}/{total_steps}")
    
    if success_steps == total_steps:
        console.print(f"[bold green]✅ 所有步骤执行成功！数据已准备好用于模型训练。[/bold green]")
        
        # 显示生成的关键文件
        data_root_path = Path(args.data_root)
        key_files = [
            data_root_path / "filtered_data.parquet",
            data_root_path / "analysis_results" / "feature_analysis.json", 
            data_root_path / "analysis_results" / "data_summary.md"
        ]
        
        console.print(f"\n[bold cyan]📁 关键输出文件:[/bold cyan]")
        for file_path in key_files:
            if file_path.exists():
                console.print(f"  ✅ {file_path}")
            else:
                console.print(f"  ❌ {file_path}")
    else:
        console.print(f"[bold red]⚠️  部分步骤执行失败，请检查错误信息。[/bold red]")

if __name__ == "__main__":
    main()

"""
使用示例:

# 完整工作流程
python scripts/examples/complete_workflow.py \
    --data_root "/path/to/data" \
    --tickers 000001.SZ 000002.SZ \
    --backend polars

# 只运行分析部分（跳过ETL）
python scripts/examples/complete_workflow.py \
    --data_root "/path/to/data" \
    --skip_etl

# 只运行ETL部分
python scripts/examples/complete_workflow.py \
    --data_root "/path/to/data" \
    --tickers 000001.SZ \
    --skip_analysis
""" 