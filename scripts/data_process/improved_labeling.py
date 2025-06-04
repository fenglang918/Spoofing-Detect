#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved Labeling for Spoofing Detection
----------------------------------------
改进的标签定义策略：
• 更细致的spoofing模式识别
• 多层次标签：快速撤单、价格操纵、订单堆积等
• 考虑市场微观结构特征
• 减少假阳性，提高标签质量
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def calculate_order_book_pressure(df):
    """计算订单簿压力指标"""
    try:
        # 检查必需的列是否存在，如果不存在则使用替代方案
        if '申买量1' in df.columns and '申卖量1' in df.columns:
            # 买卖压力不平衡
            df['book_imbalance'] = (df['申买量1'] - df['申卖量1']) / (df['申买量1'] + df['申卖量1'] + 1e-8)
        else:
            # 如果没有申买量申卖量，使用简化指标
            df['book_imbalance'] = 0.0
            print("  ⚠️ Missing 申买量1/申卖量1 columns, using simplified book_imbalance")
        
        # 价格偏离度 - 需要检查必要的列
        required_cols = ['委托价格', 'bid1', 'ask1', 'is_buy']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ⚠️ Missing columns for price_aggressiveness: {missing_cols}")
            df['price_aggressiveness'] = 0.0
        else:
            # 需要计算spread
            if 'spread' not in df.columns:
                df['spread'] = df['ask1'] - df['bid1']
            
            df['price_aggressiveness'] = np.where(
                df['is_buy'] == 1,
                (df['委托价格'] - df['bid1']) / (df['spread'] + 1e-8),
                (df['ask1'] - df['委托价格']) / (df['spread'] + 1e-8)
            )
    except Exception as e:
        print(f"  ⚠️ Error in calculate_order_book_pressure: {e}")
        # 设置默认值
        df['book_imbalance'] = 0.0
        df['price_aggressiveness'] = 0.0
    
    return df

def detect_layering_pattern(group_df):
    """检测分层下单模式（Layering）"""
    try:
        # 检查必需的列
        required_cols = ['委托_datetime', '方向_委托', '委托价格', '委托数量', 'is_buy']
        missing_cols = [col for col in required_cols if col not in group_df.columns]
        
        if missing_cols:
            print(f"    ⚠️ Missing columns for layering detection: {missing_cols}")
            group_df['layering_score'] = 0
            return group_df
        
        # 按委托时间排序
        group_df = group_df.sort_values('委托_datetime')
        
        # 检测短时间内大量同方向订单
        layering_signals = []
        
        for i, row in group_df.iterrows():
            # 检查前后1秒内的订单
            time_window = pd.Timedelta('1s')
            start_time = row['委托_datetime'] - time_window
            end_time = row['委托_datetime'] + time_window
            
            nearby_orders = group_df[
                (group_df['委托_datetime'] >= start_time) & 
                (group_df['委托_datetime'] <= end_time) &
                (group_df['方向_委托'] == row['方向_委托'])
            ]
            
            # 分层特征：多个小订单，价格递增/递减
            if len(nearby_orders) >= 3:
                prices = nearby_orders['委托价格'].values
                quantities = nearby_orders['委托数量'].values
                
                # 检查价格是否有序列性
                if row['is_buy'] == 1:
                    price_ordered = np.all(np.diff(prices) >= 0)  # 买单价格递增
                else:
                    price_ordered = np.all(np.diff(prices) <= 0)  # 卖单价格递减
                
                # 检查订单大小是否相对较小且相似
                qty_small = np.mean(quantities) < np.percentile(group_df['委托数量'], 50)
                qty_similar = np.std(quantities) / (np.mean(quantities) + 1e-8) < 0.5
                
                layering_score = int(price_ordered and qty_small and qty_similar)
            else:
                layering_score = 0
                
            layering_signals.append(layering_score)
        
        group_df['layering_score'] = layering_signals
    except Exception as e:
        print(f"    ⚠️ Error in layering detection: {e}")
        group_df['layering_score'] = 0
    
    return group_df

def improved_spoofing_rules(df):
    """改进的欺诈检测规则"""
    labels = {}
    
    try:
        # 检查必需的列并设置默认值
        if 'at_bid' not in df.columns:
            df['at_bid'] = 0
        if 'at_ask' not in df.columns:
            df['at_ask'] = 0
        if 'price_aggressiveness' not in df.columns:
            df['price_aggressiveness'] = 0.0
        if 'layering_score' not in df.columns:
            df['layering_score'] = 0
            
        # Rule 1: 快速撤单 + 市场影响
        if all(col in df.columns for col in ['存活时间_ms', '事件类型', '委托数量']):
            conditions_r1 = [
                df['存活时间_ms'] < 100,  # 快速撤单
                df['事件类型'] == '撤单',
                (df['at_bid'] == 1) | (df['at_ask'] == 1),  # 在最优价位
                df['委托数量'] > df.groupby('ticker')['委托数量'].transform('median') * 2  # 订单较大
            ]
            labels['quick_cancel_impact'] = np.all(conditions_r1, axis=0).astype(int)
        else:
            labels['quick_cancel_impact'] = np.zeros(len(df), dtype=int)
        
        # Rule 2: 价格操纵模式
        if all(col in df.columns for col in ['存活时间_ms', '事件类型', '委托数量']):
            conditions_r2 = [
                df['存活时间_ms'] < 500,
                df['事件类型'] == '撤单',
                np.abs(df['price_aggressiveness']) > 2.0,  # 价格过于激进
                df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.75)
            ]
            labels['price_manipulation'] = np.all(conditions_r2, axis=0).astype(int)
        else:
            labels['price_manipulation'] = np.zeros(len(df), dtype=int)
        
        # Rule 3: 虚假流动性提供
        if all(col in df.columns for col in ['存活时间_ms', '事件类型', '委托价格', 'bid1', 'ask1', '委托数量']):
            conditions_r3 = [
                df['存活时间_ms'] < 200,
                df['事件类型'] == '撤单',
                ((df['委托价格'] == df['bid1']) | (df['委托价格'] == df['ask1'])),
                df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', 0.9)
            ]
            labels['fake_liquidity'] = np.all(conditions_r3, axis=0).astype(int)
        else:
            labels['fake_liquidity'] = np.zeros(len(df), dtype=int)
        
        # Rule 4: 订单堆积后撤单
        if all(col in df.columns for col in ['存活时间_ms', '事件类型']):
            conditions_r4 = [
                df['layering_score'] > 0,
                df['存活时间_ms'] < 1000,
                df['事件类型'] == '撤单'
            ]
            labels['layering_cancel'] = np.all(conditions_r4, axis=0).astype(int)
        else:
            labels['layering_cancel'] = np.zeros(len(df), dtype=int)
        
        # Rule 5: 异常时间模式
        if all(col in df.columns for col in ['委托_datetime', '存活时间_ms', '事件类型', '委托数量']):
            market_active_hours = (
                (df['委托_datetime'].dt.time >= pd.to_datetime('09:30').time()) &
                (df['委托_datetime'].dt.time <= pd.to_datetime('10:30').time())
            ) | (
                (df['委托_datetime'].dt.time >= pd.to_datetime('14:00').time()) &
                (df['委托_datetime'].dt.time <= pd.to_datetime('15:00').time())
            )
            
            conditions_r5 = [
                df['存活时间_ms'] < 50,
                df['事件类型'] == '撤单',
                market_active_hours,
                df['委托数量'] > df.groupby('ticker')['委托数量'].transform('median')
            ]
            labels['active_hours_spoofing'] = np.all(conditions_r5, axis=0).astype(int)
        else:
            labels['active_hours_spoofing'] = np.zeros(len(df), dtype=int)
        
        # 综合标签：任何一种模式触发
        labels['composite_spoofing'] = (
            labels['quick_cancel_impact'] |
            labels['price_manipulation'] | 
            labels['fake_liquidity'] |
            labels['layering_cancel'] |
            labels['active_hours_spoofing']
        ).astype(int)
        
        # 保守标签：多种模式同时触发
        labels['conservative_spoofing'] = (
            (labels['quick_cancel_impact'] + 
             labels['price_manipulation'] + 
             labels['fake_liquidity'] + 
             labels['layering_cancel'] + 
             labels['active_hours_spoofing']) >= 2
        ).astype(int)
        
    except Exception as e:
        print(f"  ⚠️ Error in improved_spoofing_rules: {e}")
        # 设置所有标签为0
        n_rows = len(df)
        for label_name in ['quick_cancel_impact', 'price_manipulation', 'fake_liquidity', 
                          'layering_cancel', 'active_hours_spoofing', 'composite_spoofing', 'conservative_spoofing']:
            labels[label_name] = np.zeros(n_rows, dtype=int)
    
    return pd.DataFrame(labels)

def process_enhanced_labels(df):
    """处理增强标签"""
    print("🏷️ Creating enhanced spoofing labels...")
    
    # 1. 计算订单簿压力指标
    df = calculate_order_book_pressure(df)
    
    # 2. 按股票和日期分组处理分层检测
    enhanced_groups = []
    for (ticker, date), group in df.groupby(['ticker', '自然日']):
        print(f"  Processing {ticker} on {date}...")
        group_enhanced = detect_layering_pattern(group)
        enhanced_groups.append(group_enhanced)
    
    df_enhanced = pd.concat(enhanced_groups, ignore_index=True)
    
    # 3. 应用改进的欺诈规则
    label_df = improved_spoofing_rules(df_enhanced)
    
    # 4. 合并标签
    final_df = pd.concat([df_enhanced, label_df], axis=1)
    
    return final_df

def analyze_label_quality(df):
    """分析标签质量"""
    print("\n📊 Label Quality Analysis:")
    
    label_cols = [col for col in df.columns if 'spoofing' in col or col.startswith('quick_') 
                  or col.startswith('price_') or col.startswith('fake_') 
                  or col.startswith('layering_') or col.startswith('active_')]
    
    for col in label_cols:
        if col in df.columns:
            pos_count = df[col].sum()
            pos_rate = pos_count / len(df) * 100
            print(f"  {col}: {pos_count:,} ({pos_rate:.4f}%)")
    
    # 标签相关性分析
    if len(label_cols) > 1:
        label_corr = df[label_cols].corr()
        print(f"\n📈 Label Correlations:")
        print(label_corr.round(3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--input_pattern", default="event_stream/*/委托事件流.csv", 
                       help="输入文件模式")
    parser.add_argument("--output_dir", default="enhanced_labels", 
                       help="输出目录")
    
    args = parser.parse_args()
    
    input_root = Path(args.data_root)
    output_root = input_root / args.output_dir
    output_root.mkdir(exist_ok=True)
    
    # 查找所有事件流文件
    import glob
    pattern = str(input_root / args.input_pattern)
    event_files = glob.glob(pattern)
    
    if not event_files:
        print(f"❌ No files found matching: {pattern}")
        return
    
    print(f"🔍 Found {len(event_files)} event stream files")
    
    total_orders = 0
    total_spoofing = 0
    
    for event_file in sorted(event_files):
        date_str = Path(event_file).parent.name
        print(f"\n📅 Processing {date_str}...")
        
        try:
            # 读取事件流数据
            df = pd.read_csv(event_file, parse_dates=['委托_datetime', '事件_datetime'])
            
            if df.empty:
                print(f"  ⚠️ Empty file: {date_str}")
                continue
            
            # 处理增强标签
            df_enhanced = process_enhanced_labels(df)
            
            # 分析标签质量
            analyze_label_quality(df_enhanced)
            
            # 保存增强标签数据
            output_file = output_root / f"enhanced_labels_{date_str}.parquet"
            df_enhanced.to_parquet(output_file, index=False)
            
            # 统计
            orders_count = len(df_enhanced)
            spoofing_count = df_enhanced.get('composite_spoofing', pd.Series([0])).sum()
            
            total_orders += orders_count
            total_spoofing += spoofing_count
            
            print(f"  ✅ Saved {orders_count:,} orders, {spoofing_count} spoofing cases")
            
        except Exception as e:
            print(f"  ❌ Error processing {date_str}: {e}")
            continue
    
    print(f"\n🎯 Summary:")
    print(f"  Total orders: {total_orders:,}")
    print(f"  Total spoofing: {total_spoofing:,}")
    if total_orders > 0:
        print(f"  Spoofing rate: {total_spoofing/total_orders*100:.4f}%")

if __name__ == "__main__":
    main()

"""
# 使用示例
python scripts/data_process/improved_labeling.py \
    --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
    --input_pattern "event_stream/*/委托事件流.csv" \
    --output_dir "enhanced_labels"
""" 