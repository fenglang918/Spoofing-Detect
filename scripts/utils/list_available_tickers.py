#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
列出数据中可用的股票代码
"""

import argparse
import glob
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="列出数据中可用的股票代码")
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--date_regex", default="202503|202504|202505", help="日期正则表达式")
    parser.add_argument("--min_samples", type=int, default=100, help="最小样本数量")
    parser.add_argument("--show_stats", action="store_true", help="显示详细统计信息")
    parser.add_argument("--output_file", type=str, default=None, help="输出到文件")
    
    args = parser.parse_args()
    
    print("🔍 扫描数据中的股票代码...")
    
    # 加载特征数据
    feat_pats = [
        os.path.join(args.data_root, "features", "X_*.parquet"),
        os.path.join(args.data_root, "features_select", "X_*.parquet")  # 兼容性
    ]
    
    files = []
    for pat in feat_pats:
        files.extend(sorted(glob.glob(pat)))
    
    if not files:
        print("❌ 未找到特征数据文件")
        return
    
    print(f"📊 加载 {len(files)} 个特征文件...")
    
    # 只加载ticker和日期列以节省内存
    df_list = []
    for file in files:
        try:
            df_temp = pd.read_parquet(file, columns=['自然日', 'ticker'])
            df_list.append(df_temp)
        except Exception as e:
            print(f"⚠️ 跳过文件 {file}: {e}")
    
    if not df_list:
        print("❌ 无法读取任何数据文件")
        return
    
    df = pd.concat(df_list, ignore_index=True)
    
    # 按日期筛选
    if args.date_regex:
        date_mask = df["自然日"].astype(str).str.contains(args.date_regex)
        df = df[date_mask]
        print(f"筛选日期后数据量: {len(df):,} 条")
    
    # 统计股票信息
    ticker_stats = df.groupby('ticker').size().sort_values(ascending=False)
    
    # 按最小样本数筛选
    if args.min_samples > 0:
        ticker_stats = ticker_stats[ticker_stats >= args.min_samples]
    
    print(f"\n📋 共找到 {len(ticker_stats)} 个股票（样本数 >= {args.min_samples}）")
    
    if args.show_stats:
        print("\n📊 详细统计信息:")
        print(f"{'股票代码':<15} {'样本数':<10} {'占比%':<8}")
        print("-" * 35)
        
        total_samples = ticker_stats.sum()
        for ticker, count in ticker_stats.head(20).items():
            percentage = count / total_samples * 100
            print(f"{ticker:<15} {count:<10,} {percentage:<8.2f}")
        
        if len(ticker_stats) > 20:
            print(f"... 和其他 {len(ticker_stats) - 20} 个股票")
        
        print(f"\n总计: {total_samples:,} 条样本")
        print(f"平均每个股票: {ticker_stats.mean():.0f} 条样本")
        print(f"中位数: {ticker_stats.median():.0f} 条样本")
    else:
        # 简单列表
        print("\n可用股票代码:")
        for i, ticker in enumerate(ticker_stats.index, 1):
            print(f"{i:3d}. {ticker}")
    
    # 输出到文件
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for ticker in ticker_stats.index:
                f.write(f"{ticker}\n")
        print(f"\n✅ 股票代码已保存到: {args.output_file}")
    
    # 提供一些建议
    print(f"\n💡 使用建议:")
    print(f"1. 选择样本数较多的股票进行训练：")
    top_tickers = ticker_stats.head(5).index.tolist()
    print(f"   --include_tickers {' '.join(top_tickers)}")
    
    print(f"\n2. 限制股票数量进行快速实验：")
    print(f"   --max_tickers 10 --ticker_selection_method by_volume")
    
    print(f"\n3. 从文件读取股票列表：")
    print(f"   --ticker_file tickers.txt")

if __name__ == "__main__":
    main() 