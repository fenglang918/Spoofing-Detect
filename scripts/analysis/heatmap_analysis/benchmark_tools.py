#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heatmap Analysis Benchmark Tools
===============================
整合的性能基准测试工具集
包含：并发异常检测测试、预测热力图测试、通用性能测试
"""

import time
import numpy as np
import pandas as pd
import argparse
import multiprocessing as mp
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import joblib

# 导入核心检测函数
import sys
sys.path.append(str(Path(__file__).parent))
from manipulation_detection_heatmap import (
    ManipulationDetector, 
    parallel_statistical_features,
    parallel_anomaly_detection,
    HeatmapVisualizer,
    PerformanceMonitor
)

class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def generate_synthetic_data(self, n_samples=10000, n_stocks=50):
        """生成合成测试数据"""
        print(f"🔧 生成测试数据: {n_samples:,} 样本, {n_stocks} 股票")
        
        np.random.seed(42)
        data = {
            'ticker': np.random.choice([f'STOCK_{i:03d}' for i in range(n_stocks)], n_samples),
            'statistical_anomaly_score': np.random.exponential(0.3, n_samples),
            'order_frequency_anomaly': np.random.binomial(1, 0.1, n_samples),
            'cancel_ratio_anomaly': np.random.binomial(1, 0.05, n_samples),
            'price_volatility_anomaly': np.random.binomial(1, 0.08, n_samples),
            'qty_anomaly': np.random.binomial(1, 0.06, n_samples),
            'hour': np.random.randint(9, 16, n_samples),
            '自然日': np.random.choice(['20250301', '20250302'], n_samples),
            'composite_anomaly_score': np.random.beta(0.5, 2, n_samples),
            'is_anomalous_period': np.random.binomial(1, 0.1, n_samples),
            'known_spoofing': np.random.binomial(1, 0.05, n_samples),
            'model_spoofing_prob': np.random.beta(0.3, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        print(f"✅ 测试数据生成完成")
        return df

    def benchmark_parallel_anomaly_detection(self, df, max_workers=None):
        """并发异常检测基准测试"""
        print("\n🚀 并发异常检测基准测试")
        print("=" * 40)
        
        # 1. 传统单线程方法
        print("🔄 测试传统单线程异常检测...")
        start_time = time.time()
        detector_traditional = ManipulationDetector(enable_parallel=False)
        result_traditional = detector_traditional.detect_anomalous_periods(df.copy())
        traditional_time = time.time() - start_time
        
        # 2. 并发优化方法
        print(f"🚀 测试并发优化异常检测...")
        start_time = time.time()
        detector_parallel = ManipulationDetector(enable_parallel=True, max_workers=max_workers)
        result_parallel = detector_parallel.detect_anomalous_periods(df.copy())
        parallel_time = time.time() - start_time
        
        # 3. 计算性能提升
        speedup = traditional_time / parallel_time if parallel_time > 0 else 1
        
        print(f"✅ 传统方法: {traditional_time:.2f}s")
        print(f"✅ 并发方法: {parallel_time:.2f}s")
        print(f"📊 性能提升: {speedup:.2f}x")
        
        return {
            'traditional_time': traditional_time,
            'parallel_time': parallel_time,
            'speedup': speedup
        }

    def benchmark_heatmap_generation(self, df):
        """热力图生成性能测试"""
        print("\n📊 热力图生成性能测试")
        print("=" * 40)
        
        visualizer = HeatmapVisualizer()
        results = {}
        
        heatmap_tests = [
            ("hourly_heatmap", visualizer.create_hourly_manipulation_heatmap),
            ("daily_heatmap", visualizer.create_daily_manipulation_heatmap),
            ("prediction_comparison", visualizer.create_prediction_comparison_heatmap),
            ("correlation_heatmap", visualizer.create_manipulation_correlation_heatmap)
        ]
        
        for name, func in heatmap_tests:
            print(f"🖼️ 生成 {name}...")
            start_time = time.time()
            
            try:
                output_path = self.output_dir / f"benchmark_{name}.png"
                func(df, output_path)
                generation_time = time.time() - start_time
                results[name] = {
                    'time': generation_time,
                    'success': True,
                    'file_size': output_path.stat().st_size if output_path.exists() else 0
                }
                print(f"✅ {name}: {generation_time:.2f}s")
            except Exception as e:
                generation_time = time.time() - start_time
                results[name] = {
                    'time': generation_time,
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ {name}: 失败 - {e}")
        
        return results

    def benchmark_scaling_test(self, base_samples=5000, max_samples=30000, step=5000):
        """扩展性测试"""
        print("\n📈 扩展性测试")
        print("=" * 40)
        
        sample_sizes = list(range(base_samples, max_samples + 1, step))
        traditional_times = []
        parallel_times = []
        speedup_ratios = []
        
        for n_samples in sample_sizes:
            print(f"🧪 测试样本数: {n_samples:,}")
            
            # 生成测试数据
            df = self.generate_synthetic_data(n_samples=n_samples, n_stocks=20)
            
            # 简化的性能测试
            start_time = time.time()
            detector_trad = ManipulationDetector(enable_parallel=False)
            detector_trad.detect_anomalous_periods(df.copy())
            trad_time = time.time() - start_time
            traditional_times.append(trad_time)
            
            start_time = time.time()
            detector_par = ManipulationDetector(enable_parallel=True, max_workers=8)
            detector_par.detect_anomalous_periods(df.copy())
            par_time = time.time() - start_time
            parallel_times.append(par_time)
            
            speedup = trad_time / par_time if par_time > 0 else 1
            speedup_ratios.append(speedup)
            
            print(f"   传统: {trad_time:.2f}s, 并发: {par_time:.2f}s, 加速: {speedup:.2f}x")
        
        return sample_sizes, traditional_times, parallel_times, speedup_ratios

    def benchmark_model_prediction(self, df, batch_sizes=[1000, 5000, 10000, 20000]):
        """模型预测性能测试"""
        print("\n🤖 模型预测性能测试")
        print("=" * 40)
        
        results = {}
        
        # 创建虚拟模型进行测试
        detector = ManipulationDetector()
        
        for batch_size in batch_sizes:
            print(f"📦 测试批次大小: {batch_size:,}")
            start_time = time.time()
            
            # 模拟批量预测
            n_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
            predictions = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                
                # 模拟预测处理时间
                time.sleep(0.001 * len(batch_df) / 1000)  # 模拟计算时间
                batch_pred = np.random.random(len(batch_df))
                predictions.extend(batch_pred)
            
            prediction_time = time.time() - start_time
            throughput = len(df) / prediction_time
            
            results[batch_size] = {
                'time': prediction_time,
                'throughput': throughput,
                'n_batches': n_batches
            }
            
            print(f"✅ 批次 {batch_size:,}: {prediction_time:.2f}s, 吞吐量: {throughput:.0f} 样本/秒")
        
        return results

    def create_performance_plots(self, results):
        """创建综合性能对比图表"""
        print("\n📊 生成性能对比图表...")
        
        # 设置图表样式
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 并发异常检测性能对比
        if 'parallel_anomaly' in results:
            ax1 = plt.subplot(2, 3, 1)
            par_results = results['parallel_anomaly']
            methods = ['Traditional', 'Parallel']
            times = [par_results['traditional_time'], par_results['parallel_time']]
            
            bars = ax1.bar(methods, times, color=['red', 'green'], alpha=0.7)
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title('Anomaly Detection Performance')
            
            # 添加数值标签
            for bar, time_val in zip(bars, times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{time_val:.2f}s', ha='center', va='bottom')
        
        # 2. 扩展性测试结果
        if 'scaling' in results:
            ax2 = plt.subplot(2, 3, 2)
            sample_sizes, trad_times, par_times, speedups = results['scaling']
            
            ax2.plot(sample_sizes, speedups, 'o-', color='blue', linewidth=2, markersize=6)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Sample Size')
            ax2.set_ylabel('Speedup Ratio')
            ax2.set_title('Scaling Performance')
            ax2.grid(True, alpha=0.3)
        
        # 3. 热力图生成时间
        if 'heatmap' in results:
            ax3 = plt.subplot(2, 3, 3)
            heatmap_results = results['heatmap']
            heatmap_names = []
            heatmap_times = []
            
            for name, result in heatmap_results.items():
                if result['success']:
                    heatmap_names.append(name.replace('_', '\n'))
                    heatmap_times.append(result['time'])
            
            if heatmap_names:
                bars = ax3.bar(heatmap_names, heatmap_times, color='orange', alpha=0.7)
                ax3.set_ylabel('Generation Time (s)')
                ax3.set_title('Heatmap Generation')
                ax3.tick_params(axis='x', rotation=45)
        
        # 4. 模型预测吞吐量
        if 'model_prediction' in results:
            ax4 = plt.subplot(2, 3, 4)
            pred_results = results['model_prediction']
            batch_sizes = list(pred_results.keys())
            throughputs = [pred_results[bs]['throughput'] for bs in batch_sizes]
            
            ax4.plot(batch_sizes, throughputs, 's-', color='purple', linewidth=2, markersize=6)
            ax4.set_xlabel('Batch Size')
            ax4.set_ylabel('Throughput (samples/s)')
            ax4.set_title('Model Prediction Throughput')
            ax4.grid(True, alpha=0.3)
        
        # 5. 系统资源使用情况
        ax5 = plt.subplot(2, 3, 5)
        cpu_count = mp.cpu_count()
        memory_gb = 8  # 假设值
        
        resource_labels = ['CPU Cores', 'Memory (GB)', 'Max Workers']
        resource_values = [cpu_count, memory_gb, min(cpu_count, 20)]
        
        bars = ax5.bar(resource_labels, resource_values, color=['cyan', 'magenta', 'yellow'], alpha=0.7)
        ax5.set_ylabel('Resource Count')
        ax5.set_title('System Resources')
        
        # 6. 综合性能摘要
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
Benchmark Summary
================
CPU Cores: {cpu_count}
Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Performance Highlights:
"""
        
        if 'parallel_anomaly' in results:
            speedup = results['parallel_anomaly']['speedup']
            summary_text += f"• Anomaly Detection Speedup: {speedup:.2f}x\n"
        
        if 'scaling' in results and results['scaling'][3]:  # speedups
            max_speedup = max(results['scaling'][3])
            summary_text += f"• Max Scaling Speedup: {max_speedup:.2f}x\n"
        
        if 'heatmap' in results:
            success_count = sum(1 for r in results['heatmap'].values() if r['success'])
            total_count = len(results['heatmap'])
            summary_text += f"• Heatmap Success Rate: {success_count}/{total_count}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plot_path = self.output_dir / 'comprehensive_benchmark.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 综合性能图表已保存: {plot_path}")
        return plot_path

    def run_quick_benchmark(self, n_samples=10000, n_stocks=30):
        """快速基准测试"""
        print("🚀 快速基准测试")
        print("=" * 50)
        
        df = self.generate_synthetic_data(n_samples, n_stocks)
        results = self.benchmark_parallel_anomaly_detection(df)
        self.save_benchmark_report(results)
        return results

    def run_scaling_benchmark(self):
        """扩展性基准测试"""
        print("📈 扩展性基准测试")
        print("=" * 50)
        
        results = {}
        results['scaling'] = self.benchmark_scaling_test(5000, 25000, 5000)
        
        # 生成扩展性图表
        sample_sizes, trad_times, par_times, speedups = results['scaling']
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(sample_sizes, trad_times, 'o-', label='Traditional', linewidth=2)
        plt.plot(sample_sizes, par_times, 's-', label='Parallel', linewidth=2)
        plt.xlabel('Sample Size')
        plt.ylabel('Time (s)')
        plt.title('Execution Time vs Sample Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(sample_sizes, speedups, '^-', color='green', linewidth=2)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Sample Size')
        plt.ylabel('Speedup Ratio')
        plt.title('Parallel Speedup')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        efficiency = np.array(speedups) / mp.cpu_count()
        plt.plot(sample_sizes, efficiency, 'D-', color='orange', linewidth=2)
        plt.xlabel('Sample Size')
        plt.ylabel('Efficiency')
        plt.title(f'Parallel Efficiency (/{mp.cpu_count()} cores)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        trad_throughput = np.array(sample_sizes) / np.array(trad_times)
        par_throughput = np.array(sample_sizes) / np.array(par_times)
        plt.plot(sample_sizes, trad_throughput, 'o-', label='Traditional', linewidth=2)
        plt.plot(sample_sizes, par_throughput, 's-', label='Parallel', linewidth=2)
        plt.xlabel('Sample Size')
        plt.ylabel('Throughput (samples/s)')
        plt.title('Processing Throughput')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'scaling_benchmark.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 扩展性图表已保存: {plot_path}")
        return results

    def save_benchmark_report(self, results):
        """保存基准测试报告"""
        report_path = self.output_dir / 'benchmark_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Heatmap Analysis Benchmark Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CPU Cores: {mp.cpu_count()}\n\n")
            f.write(f"传统方法: {results['traditional_time']:.2f}s\n")
            f.write(f"并发方法: {results['parallel_time']:.2f}s\n")
            f.write(f"性能提升: {results['speedup']:.2f}x\n")
        
        print(f"📝 基准测试报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Heatmap Analysis Benchmark Tools")
    parser.add_argument("--samples", type=int, default=20000, help="测试样本数")
    parser.add_argument("--stocks", type=int, default=30, help="测试股票数")
    parser.add_argument("--output_dir", default="benchmark_results", help="输出目录")
    parser.add_argument("--scaling_test", action="store_true", help="运行扩展性测试")
    parser.add_argument("--quick_test", action="store_true", help="运行快速测试")
    parser.add_argument("--max_workers", type=int, default=None, help="最大线程数")
    
    args = parser.parse_args()
    
    # 创建基准测试套件
    benchmark = BenchmarkSuite(args.output_dir)
    
    if args.scaling_test:
        print("🚀 运行扩展性基准测试...")
        benchmark.run_scaling_benchmark()
    elif args.quick_test or not any([args.scaling_test]):
        print("🚀 运行快速基准测试...")
        benchmark.run_quick_benchmark(args.samples, args.stocks)
    
    print("\n🎉 基准测试完成！")
    print(f"📁 结果保存在: {benchmark.output_dir}")

if __name__ == "__main__":
    main()

"""
使用示例:

1. 快速基准测试:
python benchmark_tools.py --quick_test --samples 20000 --stocks 30

2. 扩展性测试:
python benchmark_tools.py --scaling_test

3. 自定义参数:
python benchmark_tools.py --samples 50000 --stocks 50 --max_workers 16

4. 指定输出目录:
python benchmark_tools.py --output_dir results/benchmarks
""" 