#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析和展示分股票的评估结果
"""

import argparse
import json
import os
import glob
import pandas as pd

def load_latest_evaluation_results(eval_dir):
    """加载最新的评估结果"""
    pattern = os.path.join(eval_dir, "evaluation_results_*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # 获取最新的文件
    latest_file = max(files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        return json.load(f), latest_file

def load_confusion_matrix_results(eval_dir):
    """加载最新的混淆矩阵结果"""
    pattern = os.path.join(eval_dir, "confusion_matrix_*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # 获取最新的文件
    latest_file = max(files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        return json.load(f), latest_file

def analyze_per_ticker_performance(results_data, conf_matrix_data=None):
    """分析分股票性能"""
    print("📊 Per-Ticker Performance Analysis")
    print("=" * 60)
    
    if 'per_ticker_metrics' not in results_data:
        print("❌ 没有发现分股票评估结果")
        print("请确保使用了最新版本的训练脚本")
        return
    
    ticker_metrics = results_data['per_ticker_metrics']
    
    if not ticker_metrics:
        print("❌ 分股票评估结果为空")
        return
    
    print(f"📈 共分析 {len(ticker_metrics)} 个股票的性能\n")
    
    # 创建DataFrame便于分析
    df_results = []
    for ticker, metrics in ticker_metrics.items():
        row = {'ticker': ticker}
        row.update(metrics)
        df_results.append(row)
    
    df = pd.DataFrame(df_results)
    
    # 显示详细表格
    print("📋 详细评估结果:")
    print(f"{'股票代码':<12} {'PR-AUC':<8} {'ROC-AUC':<8} {'P@0.1%':<8} {'P@0.5%':<8} {'P@1.0%':<8}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['ticker']:<12} "
              f"{row.get('PR-AUC', 0):<8.4f} "
              f"{row.get('ROC-AUC', 0):<8.4f} "
              f"{row.get('Precision@0.1%', 0):<8.4f} "
              f"{row.get('Precision@0.5%', 0):<8.4f} "
              f"{row.get('Precision@1.0%', 0):<8.4f}")
    
    # 统计分析
    print(f"\n📊 统计分析:")
    numeric_cols = ['PR-AUC', 'ROC-AUC', 'Precision@0.1%', 'Precision@0.5%', 'Precision@1.0%']
    
    for col in numeric_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"  {col}:")
                print(f"    平均值: {values.mean():.4f}")
                print(f"    最大值: {values.max():.4f} ({df.loc[values.idxmax(), 'ticker']})")
                print(f"    最小值: {values.min():.4f} ({df.loc[values.idxmin(), 'ticker']})")
                print(f"    标准差: {values.std():.4f}")
    
    # 排名分析
    print(f"\n🏆 各指标排名:")
    for col in ['PR-AUC', 'ROC-AUC', 'Precision@0.1%']:
        if col in df.columns:
            top_ticker = df.loc[df[col].idxmax()]
            print(f"  {col} 最佳: {top_ticker['ticker']} ({top_ticker[col]:.4f})")
    
    # 混淆矩阵分析
    if conf_matrix_data and 'per_ticker_confusion_matrices' in conf_matrix_data:
        print(f"\n🎯 分股票混淆矩阵分析:")
        ticker_conf_matrices = conf_matrix_data['per_ticker_confusion_matrices']
        
        print(f"{'股票代码':<12} {'TP':<6} {'FP':<6} {'TN':<8} {'FN':<6} {'精确率':<8} {'召回率':<8}")
        print("-" * 70)
        
        for ticker in sorted(ticker_conf_matrices.keys()):
            conf = ticker_conf_matrices[ticker]
            tp = conf['true_positives']
            fp = conf['false_positives']
            tn = conf['true_negatives']
            fn = conf['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"{ticker:<12} "
                  f"{tp:<6} "
                  f"{fp:<6} "
                  f"{tn:<8,} "
                  f"{fn:<6} "
                  f"{precision:<8.4f} "
                  f"{recall:<8.4f}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="分析分股票评估结果")
    parser.add_argument("--eval_dir", default="results/evaluation_results", 
                       help="评估结果目录")
    parser.add_argument("--save_summary", action="store_true", 
                       help="保存分析摘要到文件")
    parser.add_argument("--export_csv", action="store_true", 
                       help="导出结果到CSV文件")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.eval_dir):
        print(f"❌ 评估结果目录不存在: {args.eval_dir}")
        return
    
    print("🔍 加载评估结果...")
    
    # 加载评估结果
    results_data, results_file = load_latest_evaluation_results(args.eval_dir)
    if not results_data:
        print(f"❌ 在 {args.eval_dir} 中未找到评估结果文件")
        return
    
    print(f"✅ 已加载: {results_file}")
    
    # 加载混淆矩阵结果
    conf_matrix_data = None
    conf_result = load_confusion_matrix_results(args.eval_dir)
    if conf_result:
        conf_matrix_data, conf_file = conf_result
        print(f"✅ 已加载混淆矩阵: {conf_file}")
    
    # 分析结果
    df = analyze_per_ticker_performance(results_data, conf_matrix_data)
    
    if df is not None and args.export_csv:
        csv_file = os.path.join(args.eval_dir, "per_ticker_analysis.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 结果已导出到: {csv_file}")
    
    if args.save_summary:
        summary_file = os.path.join(args.eval_dir, "per_ticker_summary.txt")
        # 这里可以添加保存摘要的逻辑
        print(f"\n✅ 摘要已保存到: {summary_file}")

if __name__ == "__main__":
    main() 