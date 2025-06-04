#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Labeling System (No Data Leakage)
------------------------------------------
基于委托时刻可观测信息的增强标签系统
• 只使用委托提交时刻已知的信息
• 构造多种虚假报单检测规则
• 扩大正样本集合，降低预测难度
• 提供更丰富的欺诈模式识别
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def detect_suspicious_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """检测可疑价格模式（基于委托时刻信息）"""
    print("🔍 Detecting suspicious price patterns...")
    
    # 规则1: 极端价格偏离 - 远离市场价格的大单
    df['extreme_price_deviation'] = (
        (df['price_dev_prevclose'].abs() > 0.05) &  # 偏离前收盘价>5%
        (df['pct_spread'] > 5.0) &  # 远离买卖价差
        (df['log_qty'] > df.groupby('ticker')['log_qty'].transform('quantile', 0.8))  # 大单
    ).astype(int)
    
    # 规则2: 价格激进性 - 穿越买卖价差的订单
    df['aggressive_pricing'] = (
        ((df['is_buy'] == 1) & (df['delta_mid'] > df['spread'] * 0.5)) |  # 买单价格过高
        ((df['is_buy'] == 0) & (df['delta_mid'] < -df['spread'] * 0.5))   # 卖单价格过低
    ).astype(int)
    
    # 规则3: 异常价格层级 - 精确在买一卖一价位的大单
    df['at_touch_large_order'] = (
        (((df['委托价格'] == df['bid1']) & (df['方向_委托'] == '买')) |
         ((df['委托价格'] == df['ask1']) & (df['方向_委托'] == '卖'))) &
        (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.9))
    ).astype(int)
    
    return df

def detect_timing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """检测时间模式异常（基于委托时刻信息）"""
    print("⏰ Detecting timing patterns...")
    
    # 规则4: 市场活跃时段异常 - 开盘/收盘时段的异常行为
    market_open = ((df['委托_datetime'].dt.time >= pd.to_datetime('09:30').time()) & 
                   (df['委托_datetime'].dt.time <= pd.to_datetime('09:45').time()))
    market_close = ((df['委托_datetime'].dt.time >= pd.to_datetime('14:45').time()) & 
                    (df['委托_datetime'].dt.time <= pd.to_datetime('15:00').time()))
    
    df['volatile_period_anomaly'] = (
        (market_open | market_close) &
        (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('median') * 3) &
        (df['pct_spread'] > 2.0)
    ).astype(int)
    
    # 规则5: 集中下单模式 - 短时间内大量订单
    df['burst_order_pattern'] = (
        (df['orders_100ms'] > 5) &  # 100ms内超过5笔订单
        (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.7))
    ).astype(int)
    
    return df

def detect_quantity_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """检测数量模式异常（基于委托时刻信息）"""
    print("📊 Detecting quantity patterns...")
    
    # 规则6: 异常大单 - 远超正常交易规模
    df['abnormal_large_order'] = (
        df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.95)
    ).astype(int)
    
    # 规则7: 整数倍异常 - 可疑的整数倍数量
    df['round_number_pattern'] = (
        ((df['委托数量'] % 10000 == 0) & (df['委托数量'] >= 50000)) |  # 大额整万
        ((df['委托数量'] % 1000 == 0) & (df['委托数量'] >= 10000))    # 中额整千
    ).astype(int)
    
    return df

def detect_market_microstructure_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """检测市场微观结构异常（基于委托时刻信息）"""
    print("🏗️ Detecting market microstructure patterns...")
    
    # 规则8: 订单簿不平衡利用 - 利用买卖失衡的订单
    df['imbalance_exploitation'] = (
        (df['book_imbalance'].abs() > 0.5) &  # 强烈的买卖不平衡
        (((df['book_imbalance'] > 0) & (df['方向_委托'] == '卖')) |  # 买盘强势时卖出
         ((df['book_imbalance'] < 0) & (df['方向_委托'] == '买'))) &  # 卖盘强势时买入
        (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('median') * 2)
    ).astype(int)
    
    # 规则9: 价差操纵迹象 - 可能影响价差的订单
    df['spread_manipulation_signal'] = (
        (df['spread'] > 0) &
        (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.8)) &
        (((df['方向_委托'] == '买') & (df['委托价格'] >= df['ask1'] - df['spread'] * 0.1)) |
         ((df['方向_委托'] == '卖') & (df['委托价格'] <= df['bid1'] + df['spread'] * 0.1)))
    ).astype(int)
    
    return df

def detect_behavioral_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """检测行为模式异常（基于历史观测信息）"""
    print("🎭 Detecting behavioral patterns...")
    
    # 规则10: 频繁撤单历史 - 基于过去的撤单行为
    df['frequent_canceller'] = (
        df['cancels_5s'] > 3  # 过去5秒内撤单超过3次
    ).astype(int)
    
    # 规则11: 拍板意愿异常 - 价格激进但可能是虚假意图
    df['false_aggressive_intent'] = (
        (df['price_aggressiveness'] > 1.0) &  # 价格激进
        (df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.75)) &
        (df['orders_100ms'] > 2)  # 短时间内多笔订单
    ).astype(int)
    
    return df

def create_enhanced_labels(df: pd.DataFrame) -> pd.DataFrame:
    """创建综合的增强标签"""
    print("\n🏷️ Creating enhanced spoofing labels...")
    
    # 应用所有检测规则
    df = detect_suspicious_price_patterns(df)
    df = detect_timing_patterns(df)  
    df = detect_quantity_patterns(df)
    df = detect_market_microstructure_patterns(df)
    df = detect_behavioral_patterns(df)
    
    # 单项规则标签
    pattern_rules = [
        'extreme_price_deviation', 'aggressive_pricing', 'at_touch_large_order',
        'volatile_period_anomaly', 'burst_order_pattern', 'abnormal_large_order', 
        'round_number_pattern', 'imbalance_exploitation', 'spread_manipulation_signal',
        'frequent_canceller', 'false_aggressive_intent'
    ]
    
    # 综合标签1：任意规则触发（宽松）
    df['enhanced_spoofing_liberal'] = (
        df[pattern_rules].sum(axis=1) >= 1
    ).astype(int)
    
    # 综合标签2：多个规则触发（中等）
    df['enhanced_spoofing_moderate'] = (
        df[pattern_rules].sum(axis=1) >= 2
    ).astype(int)
    
    # 综合标签3：严格规则（保守）
    df['enhanced_spoofing_strict'] = (
        df[pattern_rules].sum(axis=1) >= 3
    ).astype(int)
    
    # 分类标签：基于最强信号
    max_signals = df[pattern_rules].sum(axis=1)
    df['pattern_strength'] = max_signals
    
    # 高质量标签：结合原始规则 + 增强模式
    if 'y_label' in df.columns:
        df['enhanced_combined'] = (
            (df['y_label'] == 1) |  # 原始规则
            (df['enhanced_spoofing_moderate'] == 1)  # 或增强规则
        ).astype(int)
    else:
        df['enhanced_combined'] = df['enhanced_spoofing_moderate']
    
    return df

def analyze_enhanced_labels(df: pd.DataFrame):
    """分析增强标签质量"""
    print("\n📊 Enhanced Label Analysis:")
    
    label_cols = [col for col in df.columns if 'enhanced' in col or 'pattern' in col or 
                  col in ['extreme_price_deviation', 'aggressive_pricing', 'at_touch_large_order',
                          'volatile_period_anomaly', 'burst_order_pattern', 'abnormal_large_order']]
    
    for col in label_cols:
        if col in df.columns:
            pos_count = df[col].sum()
            pos_rate = pos_count / len(df) * 100 if len(df) > 0 else 0
            print(f"  {col:<30}: {pos_count:>8,} ({pos_rate:>6.3f}%)")
    
    # 对比原始标签
    if 'y_label' in df.columns:
        original_pos = df['y_label'].sum()
        enhanced_pos = df['enhanced_combined'].sum() if 'enhanced_combined' in df.columns else 0
        print(f"\n📈 Label Comparison:")
        print(f"  Original y_label:           {original_pos:>8,} ({original_pos/len(df)*100:>6.3f}%)")
        print(f"  Enhanced combined:          {enhanced_pos:>8,} ({enhanced_pos/len(df)*100:>6.3f}%)")
        if original_pos > 0:
            improvement = (enhanced_pos - original_pos) / original_pos * 100
            print(f"  Improvement:                {enhanced_pos-original_pos:>8,} ({improvement:>+6.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--output_suffix", default="_enhanced", help="输出文件后缀")
    parser.add_argument("--train_regex", default="202503|202504", help="训练数据日期正则")
    parser.add_argument("--valid_regex", default="202505", help="验证数据日期正则")
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Labeling System (No Data Leakage)")
    print("=" * 60)
    
    # 读取特征数据
    feat_dir = Path(args.data_root) / "features_select"
    feat_files = list(feat_dir.glob("X_*.parquet"))
    
    if not feat_files:
        print("❌ No feature files found")
        return
    
    print(f"📁 Found {len(feat_files)} feature files")
    
    # 读取标签数据  
    label_dir = Path(args.data_root) / "labels_select"
    label_files = list(label_dir.glob("labels_*.parquet"))
    
    print(f"📁 Found {len(label_files)} label files")
    
    # 合并数据
    df_features = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    df_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
    
    print(f"📊 Features shape: {df_features.shape}")
    print(f"📊 Labels shape: {df_labels.shape}")
    
    # 合并
    df = df_features.merge(df_labels, on=['自然日', 'ticker', '交易所委托号'], how='inner')
    print(f"📊 Merged shape: {df.shape}")
    
    # 数据预处理
    df['委托_datetime'] = pd.to_datetime(df['自然日'].astype(str).str[:8], format='%Y%m%d')
    
    # 填充缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 创建增强标签
    df = create_enhanced_labels(df)
    
    # 分析结果
    analyze_enhanced_labels(df)
    
    # 保存增强标签结果
    output_dir = Path(args.data_root) / f"labels_enhanced{args.output_suffix}"
    output_dir.mkdir(exist_ok=True)
    
    # 按日期保存
    label_cols = ['自然日', 'ticker', '交易所委托号', 'y_label'] + [
        col for col in df.columns if 'enhanced' in col or 'pattern' in col or
        col in ['extreme_price_deviation', 'aggressive_pricing', 'at_touch_large_order',
                'volatile_period_anomaly', 'burst_order_pattern', 'abnormal_large_order',
                'round_number_pattern', 'imbalance_exploitation', 'spread_manipulation_signal',
                'frequent_canceller', 'false_aggressive_intent']
    ]
    
    label_cols = [col for col in label_cols if col in df.columns]
    
    for date in df['自然日'].unique():
        date_data = df[df['自然日'] == date][label_cols]
        output_file = output_dir / f"enhanced_labels_{date}.parquet"
        date_data.to_parquet(output_file, index=False)
        print(f"💾 Saved: {output_file}")
    
    print(f"\n✅ Enhanced labeling completed!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

"""
使用示例：

# 创建增强标签
python scripts/data_process/enhanced_labeling_no_leakage.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data"

# 创建更严格的增强标签
python scripts/data_process/enhanced_labeling_no_leakage.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --output_suffix "_strict"
""" 