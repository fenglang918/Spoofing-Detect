#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据泄露诊断工具
================
分析当前数据集中的潜在数据泄露问题，并提供修复建议
"""

import argparse
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

def load_data(data_root):
    """加载特征和标签数据"""
    feat_pats = [os.path.join(data_root, "features_select", "X_*.parquet")]
    lab_pats = [os.path.join(data_root, "labels_select", "labels_*.parquet")]
    
    feat_files = []
    for pat in feat_pats:
        feat_files.extend(sorted(glob.glob(pat)))
    
    lab_files = []
    for pat in lab_pats:
        lab_files.extend(sorted(glob.glob(pat)))
        
    if not feat_files or not lab_files:
        raise FileNotFoundError("找不到特征或标签文件")
    
    console.print(f"发现 {len(feat_files)} 个特征文件和 {len(lab_files)} 个标签文件")
    
    df_feat = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    df_lab = pd.concat([pd.read_parquet(f) for f in lab_files], ignore_index=True)
    
    # 合并数据
    df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="inner")
    
    return df

def identify_leakage_features(df):
    """识别可能包含数据泄露的特征"""
    
    leakage_features = {
        "直接未来信息": [
            "存活时间_ms", "final_survival_time_ms"
        ],
        "成交相关（委托时刻未知）": [
            "total_traded_qty", "num_trades", "is_fully_filled"
        ],
        "聚合统计（包含未来事件）": [
            "total_events", "num_cancels"
        ],
        "撤单标志（委托时刻未知）": [
            "is_cancel"
        ],
        "其他可疑特征": [
            "cancellation_flag", "fill_ratio", "execution_time"
        ]
    }
    
    safe_features = [
        # 行情特征（委托时刻可观测）
        "bid1", "ask1", "prev_close", "mid_price", "spread",
        
        # 价格特征（委托时刻可观测）
        "delta_mid", "pct_spread", "price_dev_prevclose",
        
        # 订单特征（委托时刻可观测）
        "is_buy", "log_qty", "委托价格", "委托数量",
        
        # 历史统计特征（只使用过去的信息）
        "orders_100ms", "cancels_5s", 
        
        # 时间特征
        "time_sin", "time_cos", "in_auction"
    ]
    
    return leakage_features, safe_features

def analyze_feature_target_correlation(df, feature, target="y_label"):
    """分析特征与标签的相关性（用于检测泄露）"""
    if feature not in df.columns:
        return None
    
    # 计算相关性
    correlation = df[feature].corr(df[target])
    
    # 计算互信息（对于分类变量）
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(df[[feature]].fillna(0), df[target], random_state=42)[0]
    except:
        mi = None
    
    return {
        'correlation': correlation,
        'mutual_info': mi,
        'unique_values': df[feature].nunique(),
        'missing_rate': df[feature].isnull().mean()
    }

def temporal_analysis(df):
    """时间序列分析"""
    console.print("\n🕒 时间序列分析")
    
    # 按日期统计
    date_stats = df.groupby('自然日').agg({
        'y_label': ['count', 'sum', 'mean']
    }).round(4)
    
    date_stats.columns = ['总样本数', '正样本数', '正样本率']
    
    table = Table(title="按日期统计")
    table.add_column("日期", style="cyan")
    table.add_column("总样本数", style="magenta")
    table.add_column("正样本数", style="yellow")
    table.add_column("正样本率", style="green")
    
    for date, row in date_stats.iterrows():
        table.add_row(
            str(date),
            f"{int(row['总样本数']):,}",
            f"{int(row['正样本数']):,}",
            f"{row['正样本率']:.4f}"
        )
    
    console.print(table)
    
    return date_stats

def feature_importance_analysis(df, target="y_label"):
    """特征重要性分析（基于简单相关性）"""
    console.print("\n📊 特征重要性分析")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ["自然日", "y_label"]]
    
    correlations = []
    for col in feature_cols:
        corr = analyze_feature_target_correlation(df, col, target)
        if corr:
            correlations.append({
                'feature': col,
                'correlation': abs(corr['correlation']),
                'mutual_info': corr['mutual_info'] or 0,
                'unique_values': corr['unique_values'],
                'missing_rate': corr['missing_rate']
            })
    
    # 排序特征
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    # 显示top特征
    table = Table(title="特征-标签相关性 Top 15")
    table.add_column("特征", style="cyan")
    table.add_column("相关性", style="magenta")
    table.add_column("唯一值数", style="yellow")
    table.add_column("缺失率", style="green")
    
    for _, row in corr_df.head(15).iterrows():
        table.add_row(
            row['feature'],
            f"{row['correlation']:.4f}",
            f"{int(row['unique_values']):,}",
            f"{row['missing_rate']:.4f}"
        )
    
    console.print(table)
    
    return corr_df

def detect_perfect_predictors(df, target="y_label", threshold=0.95):
    """检测完美预测特征（可能的数据泄露）"""
    console.print(f"\n🚨 检测完美预测特征（相关性 > {threshold}）")
    
    leakage_features, _ = identify_leakage_features(df)
    all_leakage = []
    for category, features in leakage_features.items():
        all_leakage.extend(features)
    
    perfect_predictors = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == target:
            continue
        
        corr = df[col].corr(df[target])
        if abs(corr) > threshold:
            is_known_leakage = col in all_leakage
            perfect_predictors.append({
                'feature': col,
                'correlation': corr,
                'is_known_leakage': is_known_leakage
            })
    
    if perfect_predictors:
        table = Table(title="🚨 完美预测特征（疑似数据泄露）")
        table.add_column("特征", style="cyan")
        table.add_column("相关性", style="red")
        table.add_column("已知泄露", style="yellow")
        
        for pred in perfect_predictors:
            table.add_row(
                pred['feature'],
                f"{pred['correlation']:.6f}",
                "✅" if pred['is_known_leakage'] else "❓"
            )
        
        console.print(table)
    else:
        console.print("✅ 未发现完美预测特征")
    
    return perfect_predictors

def analyze_survival_time_distribution(df):
    """分析存活时间分布"""
    if "存活时间_ms" not in df.columns and "final_survival_time_ms" not in df.columns:
        console.print("⚠️ 未找到存活时间特征")
        return
    
    console.print("\n⏱️ 存活时间分析")
    
    survival_col = "存活时间_ms" if "存活时间_ms" in df.columns else "final_survival_time_ms"
    
    # 按标签分组统计存活时间
    survival_stats = df.groupby('y_label')[survival_col].describe()
    
    console.print("存活时间统计（按标签分组）:")
    console.print(survival_stats)
    
    # 计算不同阈值下的标签分布
    thresholds = [50, 100, 500, 1000, 2000, 5000]
    
    table = Table(title="存活时间阈值分析")
    table.add_column("阈值(ms)", style="cyan")
    table.add_column("< 阈值样本数", style="magenta")
    table.add_column("< 阈值正样本率", style="yellow")
    table.add_column(">= 阈值样本数", style="green")
    table.add_column(">= 阈值正样本率", style="blue")
    
    for threshold in thresholds:
        below_threshold = df[df[survival_col] < threshold]
        above_threshold = df[df[survival_col] >= threshold]
        
        below_pos_rate = below_threshold['y_label'].mean() if len(below_threshold) > 0 else 0
        above_pos_rate = above_threshold['y_label'].mean() if len(above_threshold) > 0 else 0
        
        table.add_row(
            str(threshold),
            f"{len(below_threshold):,}",
            f"{below_pos_rate:.4f}",
            f"{len(above_threshold):,}",
            f"{above_pos_rate:.4f}"
        )
    
    console.print(table)

def generate_recommendations(df):
    """生成修复建议"""
    console.print("\n💡 修复建议")
    
    leakage_features, safe_features = identify_leakage_features(df)
    
    # 检查当前数据集中存在的泄露特征
    existing_leakage = []
    for category, features in leakage_features.items():
        existing = [f for f in features if f in df.columns]
        if existing:
            existing_leakage.extend(existing)
    
    if existing_leakage:
        console.print("🚫 [bold red]发现以下数据泄露特征，必须移除：[/bold red]")
        for feat in existing_leakage:
            console.print(f"   • {feat}")
    
    # 检查安全特征
    existing_safe = [f for f in safe_features if f in df.columns]
    console.print(f"\n✅ [bold green]可以安全使用的特征 ({len(existing_safe)} 个)：[/bold green]")
    for feat in existing_safe:
        console.print(f"   • {feat}")
    
    missing_safe = [f for f in safe_features if f not in df.columns]
    if missing_safe:
        console.print(f"\n⚠️ [bold yellow]建议添加的安全特征：[/bold yellow]")
        for feat in missing_safe:
            console.print(f"   • {feat}")
    
    console.print("\n📋 [bold cyan]具体修复步骤：[/bold cyan]")
    console.print("1. 重新运行特征工程，移除所有包含未来信息的特征")
    console.print("2. 确保标签构造只基于委托后的真实结果（不是预测结果）")
    console.print("3. 实施严格的时间序列验证（训练集日期 < 验证集日期）")
    console.print("4. 添加正则化和早停，防止过拟合")
    console.print("5. 使用我提供的修复版训练脚本: scripts/train/train_baseline_fixed.py")

def main():
    parser = argparse.ArgumentParser(description="数据泄露诊断工具")
    parser.add_argument("--data_root", required=True, 
                      help="数据根目录")
    parser.add_argument("--output_dir", default="./leakage_analysis", 
                      help="输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    
    console.print("🔍 [bold blue]开始数据泄露诊断...[/bold blue]")
    
    # 加载数据
    try:
        df = load_data(args.data_root)
        console.print(f"✅ 成功加载数据：{df.shape}")
    except Exception as e:
        console.print(f"❌ 数据加载失败：{e}")
        return
    
    # 基本统计
    console.print(f"\n📈 数据概览：")
    console.print(f"   • 总样本数：{len(df):,}")
    console.print(f"   • 特征数：{len(df.columns) - 4}")  # 减去ID列
    console.print(f"   • 正样本数：{df['y_label'].sum():,}")
    console.print(f"   • 正样本率：{df['y_label'].mean():.6f}")
    
    # 时间序列分析
    date_stats = temporal_analysis(df)
    
    # 特征重要性分析
    corr_df = feature_importance_analysis(df)
    
    # 检测完美预测特征
    perfect_predictors = detect_perfect_predictors(df)
    
    # 存活时间分析
    analyze_survival_time_distribution(df)
    
    # 生成建议
    generate_recommendations(df)
    
    # 保存分析结果
    try:
        corr_df.to_csv(Path(args.output_dir) / "feature_correlations.csv", index=False)
        date_stats.to_csv(Path(args.output_dir) / "temporal_stats.csv")
        console.print(f"\n💾 分析结果已保存到：{args.output_dir}")
    except Exception as e:
        console.print(f"⚠️ 保存失败：{e}")
    
    console.print("\n🎯 [bold green]诊断完成！请查看上述建议并使用修复版训练脚本。[/bold green]")

if __name__ == "__main__":
    main()

"""
# 使用示例
python scripts/analyze_data_leakage.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --output_dir "./leakage_analysis"
""" 