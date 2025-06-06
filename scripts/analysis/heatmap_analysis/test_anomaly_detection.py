#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异常检测算法测试脚本
用于诊断manipulation_detection_heatmap.py中的卡死问题
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import psutil
import gc

def test_single_algorithm(algorithm_name, algorithm, X, timeout=60):
    """测试单个算法"""
    print(f"\n🧪 测试 {algorithm_name} 算法...")
    print(f"   数据形状: {X.shape}")
    print(f"   内存使用: {X.nbytes / 1024 / 1024:.1f}MB")
    
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    
    try:
        if algorithm_name == 'local_outlier_factor':
            result = algorithm.fit_predict(X)
        elif algorithm_name == 'dbscan':
            result = algorithm.fit_predict(X)
        else:  # isolation_forest
            result = algorithm.fit_predict(X)
        
        elapsed = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory
        
        anomaly_count = (result == -1).sum() if hasattr(result, 'sum') else 0
        
        print(f"✅ {algorithm_name} 成功完成")
        print(f"   耗时: {elapsed:.2f}s")
        print(f"   内存增加: {memory_used:.1f}MB")
        print(f"   异常点数: {anomaly_count}")
        
        return True, elapsed, memory_used, anomaly_count
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {algorithm_name} 失败: {e}")
        print(f"   耗时: {elapsed:.2f}s")
        return False, elapsed, 0, 0

def main():
    """主测试函数"""
    print("🧪 异常检测算法诊断测试")
    print("=" * 50)
    
    # 生成测试数据 - 不同规模
    test_sizes = [1000, 5000, 10000, 25000, 50000]
    
    for size in test_sizes:
        print(f"\n📊 测试数据规模: {size:,} 行 x 10 特征")
        
        # 生成随机数据
        np.random.seed(42)
        X = np.random.randn(size, 10)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 测试参数
        contamination = 0.1
        min_samples = 5
        
        # 定义算法
        algorithms = {
            'isolation_forest': IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=1,
                max_samples=min(1000, len(X))
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=min(min_samples, 10),
                n_jobs=1
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=min(20, len(X)-1),
                n_jobs=1
            )
        }
        
        # 测试每个算法
        results = {}
        for name, algo in algorithms.items():
            success, elapsed, memory, anomaly_count = test_single_algorithm(name, algo, X_scaled)
            results[name] = {
                'success': success,
                'elapsed': elapsed,
                'memory': memory,
                'anomaly_count': anomaly_count
            }
            
            # 清理内存
            gc.collect()
            
            # 如果算法失败或超时，跳出循环
            if not success or elapsed > 120:  # 2分钟超时
                print(f"⚠️ {name} 算法在数据规模 {size:,} 时出现问题，停止测试更大规模")
                break
        
        # 打印该规模的总结
        print(f"\n📋 数据规模 {size:,} 的测试结果:")
        for name, result in results.items():
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {name}: {result['elapsed']:.1f}s, {result['memory']:.1f}MB, {result['anomaly_count']} 异常")
        
        # 如果有算法失败，停止测试更大规模
        if not all(result['success'] for result in results.values()):
            print(f"\n⚠️ 在数据规模 {size:,} 时发现问题，建议:")
            print(f"   1. 使用 --sample_size {size//2} 参数限制数据规模")
            print(f"   2. 使用 --fast_mode 参数仅使用IsolationForest")
            print(f"   3. 增加系统内存或使用更小的批次大小")
            break
    
    print("\n🎯 测试完成！")
    print("💡 根据测试结果调整参数:")
    print("   - 如果LocalOutlierFactor卡死：添加 --fast_mode")
    print("   - 如果DBSCAN很慢：减少 --min_samples")
    print("   - 如果内存不足：减少 --sample_size")

if __name__ == "__main__":
    main() 