#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Optimization Pipeline for Spoofing Detection
-----------------------------------------------------
自动化运行完整的优化流程：
1. 增强特征工程和多种模型训练对比（假设已运行ETL生成增强标签）
2. 性能分析和报告生成

前置条件：
需要先运行 ETL 流程生成数据和标签：
python scripts/data_process/run_etl_from_event.py \
    --root "/path/to/event_stream" \
    --enhanced_labels  # 生成增强标签

使用方法：
python run_optimization_pipeline.py --data_root "/path/to/data"
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
import json

def run_command(cmd, description):
    """运行命令并处理结果"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time
        print(f"✅ {description} completed successfully in {elapsed:.1f}s")
        
        if result.stdout:
            print(f"Output preview:\n{result.stdout[:500]}...")
        
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.1f}s")
        print(f"Error: {e.stderr}")
        return False, e.stderr

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'lightgbm', 
        'xgboost', 'optuna', 'imblearn', 
        'matplotlib', 'seaborn'
    ]
    
    # 包名映射（import名 -> pip安装名）
    package_mapping = {
        'sklearn': 'scikit-learn',
        'imblearn': 'imbalanced-learn'
    }
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            # 使用pip安装名
            pip_name = package_mapping.get(package, package)
            missing.append(pip_name)
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def check_data_availability(data_root, use_enhanced_labels):
    """检查数据可用性"""
    print("🔍 Checking data availability...")
    
    data_path = Path(data_root)
    features_dir = data_path / "features_select"
    labels_dir = data_path / "labels_select"
    
    if not features_dir.exists():
        print(f"❌ Features directory not found: {features_dir}")
        print("Please run ETL pipeline first:")
        print("python scripts/data_process/run_etl_from_event.py --root /path/to/event_stream --enhanced_labels")
        return False
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        print("Please run ETL pipeline first:")
        print("python scripts/data_process/run_etl_from_event.py --root /path/to/event_stream --enhanced_labels")
        return False
    
    # 检查文件数量
    feature_files = list(features_dir.glob("X_*.parquet"))
    label_files = list(labels_dir.glob("labels_*.parquet"))
    
    print(f"Found {len(feature_files)} feature files and {len(label_files)} label files")
    
    if len(feature_files) == 0 or len(label_files) == 0:
        print("❌ No data files found. Please run ETL pipeline first.")
        return False
    
    # 如果使用增强标签，检查标签文件中是否包含增强标签列
    if use_enhanced_labels and len(label_files) > 0:
        try:
            import pandas as pd
            sample_labels = pd.read_parquet(label_files[0])
            enhanced_cols = ['composite_spoofing', 'conservative_spoofing', 'quick_cancel_impact', 
                           'price_manipulation', 'fake_liquidity', 'layering_cancel']
            
            available_enhanced = [col for col in enhanced_cols if col in sample_labels.columns]
            if not available_enhanced:
                print("⚠️ Enhanced labels requested but not found in data.")
                print("Please re-run ETL with --enhanced_labels flag:")
                print("python scripts/data_process/run_etl_from_event.py --root /path/to/event_stream --enhanced_labels")
                return False
            else:
                print(f"✅ Found enhanced label columns: {available_enhanced}")
        except Exception as e:
            print(f"⚠️ Could not verify enhanced labels: {e}")
    
    print("✅ Data availability check passed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run complete optimization pipeline")
    parser.add_argument("--data_root", required=True, 
                       help="Root directory containing features_select and labels_select")
    parser.add_argument("--results_dir", 
                       help="Directory to save all results and analysis output (default: {data_root}/results)")
    parser.add_argument("--skip_baseline", action="store_true", 
                       help="Skip baseline training")
    parser.add_argument("--use_enhanced_labels", action="store_true",
                       help="Use enhanced spoofing labels instead of original y_label")
    parser.add_argument("--label_type", choices=["composite_spoofing", "conservative_spoofing"], 
                       default="composite_spoofing",
                       help="Which enhanced label type to use (only with --use_enhanced_labels)")
    parser.add_argument("--experiments", nargs="+", 
                       default=["enhanced", "ensemble", "optimized"],
                       help="Experiments to run")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ Data root does not exist: {data_root}")
        return
    
    # 设置结果保存目录
    if args.results_dir:
        results_base_dir = Path(args.results_dir)
    else:
        results_base_dir = data_root / "results"
    
    # 创建结果目录结构
    results_base_dir.mkdir(exist_ok=True)
    experiment_results_dir = results_base_dir / "experiment_results"
    analysis_output_dir = results_base_dir / "analysis_output"
    experiment_results_dir.mkdir(exist_ok=True)
    analysis_output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Results will be saved to: {results_base_dir}")
    if args.use_enhanced_labels:
        print(f"🏷️ Using enhanced labels: {args.label_type}")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查数据可用性
    if not check_data_availability(data_root, args.use_enhanced_labels):
        return
    
    results = {}
    total_start = time.time()
    
    print(f"\n🎯 Starting optimization pipeline for: {data_root}")
    print(f"Experiments to run: {args.experiments}")
    
    # Step 1: 基线模型训练（可选）
    if not args.skip_baseline:
        success, output = run_command([
            sys.executable, "scripts/train/train_baseline_fixed.py",
            "--data_root", str(data_root),
            "--train_regex", "202503|202504",
            "--valid_regex", "202505",
            "--device", "cpu",
            "--balance_method", "undersample"
        ], "Step 1: Training baseline model")
        
        if success:
            # 解析基线结果
            try:
                # 从输出中提取关键指标
                lines = output.split('\n')
                pr_auc = None
                prec_k = None
                
                for line in lines:
                    if "PR-AUC：" in line:
                        pr_auc = float(line.split('：')[1])
                    elif "Precision@Top0.1%：" in line:
                        prec_k = float(line.split('：')[1])
                
                results['baseline'] = {
                    'PR-AUC': pr_auc,
                    'Precision@0.1%': prec_k,
                    'method': 'LightGBM + Undersample'
                }
            except:
                print("⚠️ Could not parse baseline results")
    
    # Step 2: 运行增强实验
    experiment_configs = {
        'enhanced': {
            'args': ['--sampling_method', 'undersample'],
            'description': 'Enhanced features + Undersample'
        },
        'ensemble': {
            'args': ['--use_ensemble', '--sampling_method', 'undersample'],
            'description': 'Model ensemble + Undersample'
        },
        'optimized': {
            'args': ['--optimize_params', '--sampling_method', 'undersample', '--n_trials', '30'],
            'description': 'Hyperparameter optimization + Undersample'
        }
    }
    
    for exp_name in args.experiments:
        if exp_name not in experiment_configs:
            print(f"⚠️ Unknown experiment: {exp_name}")
            continue
        
        config = experiment_configs[exp_name]
        
        cmd = [
            sys.executable, "scripts/train/train_baseline_enhanced_fixed.py",
            "--data_root", str(data_root),
            "--train_regex", "202503|202504",
            "--valid_regex", "202505"
        ] + config['args']
        
        # 添加增强标签参数
        if args.use_enhanced_labels:
            cmd.extend(["--use_enhanced_labels", "--label_type", args.label_type])
        
        success, output = run_command(cmd, f"Step 2.{exp_name}: {config['description']}")
        
        if success:
            # 解析结果（改进版）
            try:
                # 从输出中提取关键指标
                lines = output.split('\n')
                pr_auc = None
                prec_k = None
                roc_auc = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("PR-AUC:"):
                        pr_auc = float(line.split(':')[1].strip())
                    elif line.startswith("Precision@0.1%:"):
                        prec_k = float(line.split(':')[1].strip())
                    elif line.startswith("ROC-AUC:"):
                        roc_auc = float(line.split(':')[1].strip())
                
                label_suffix = f"_{args.label_type}" if args.use_enhanced_labels else "_original"
                results[exp_name] = {
                    'method': config['description'] + label_suffix,
                    'PR-AUC': pr_auc,
                    'Precision@0.1%': prec_k,
                    'ROC-AUC': roc_auc,
                    'success': True
                }
                
                # 显示解析结果
                if pr_auc is not None:
                    print(f"  📊 Parsed results: PR-AUC={pr_auc:.6f}")
                else:
                    print(f"  ⚠️ Could not parse PR-AUC from output")
                    
            except Exception as e:
                print(f"⚠️ Could not parse {exp_name} results: {e}")
                results[exp_name] = {
                    'method': config['description'],
                    'success': False,
                    'error': str(e)
                }
    
    # Step 3: 性能对比和分析
    if len(results) > 1:
        # 保存结果为JSON
        for exp_name, result in results.items():
            result_file = experiment_results_dir / f"{exp_name}_results.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # 运行性能对比
        success, output = run_command([
            sys.executable, "scripts/analysis/performance_comparison.py",
            "--results_dir", str(experiment_results_dir),
            "--output_dir", str(analysis_output_dir)
        ], "Step 3: Performance analysis and comparison")
    
    # 总结
    total_elapsed = time.time() - total_start
    print(f"\n🎉 Pipeline completed in {total_elapsed/60:.1f} minutes")
    print(f"📊 Results summary:")
    
    for exp_name, result in results.items():
        status = "✅" if result.get('success', False) else "❌"
        method = result.get('method', 'Unknown')
        pr_auc = result.get('PR-AUC', 'N/A')
        print(f"  {status} {exp_name}: {method} (PR-AUC: {pr_auc})")
    
    # 优化建议
    print(f"\n💡 Next steps:")
    print(f"  1. Review analysis results in: {analysis_output_dir}")
    print(f"  2. Check detailed logs for each experiment")
    print(f"  3. Consider running additional experiments based on recommendations")
    
    if len(results) < 2:
        print(f"  4. Run more experiments for better comparison")

if __name__ == "__main__":
    main()

"""
使用示例：

# 前置步骤：运行ETL生成增强标签
python scripts/data_process/run_etl_from_event.py \
  --root "/obs/users/fenglang/general/Spoofing Detect/data/event_stream" \
  --enhanced_labels \
  --backend polars

# 运行完整优化流程，指定结果保存目录
python run_optimization_pipeline.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --results_dir "/home/ma-user/code/fenglang/Spoofing Detect/results"

# 使用增强标签（ETL流程已生成）
python run_optimization_pipeline.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --results_dir "/home/ma-user/code/fenglang/Spoofing Detect/results" \
  --use_enhanced_labels \
  --label_type "composite_spoofing"

# 使用保守标签（更严格的欺诈检测）
python run_optimization_pipeline.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --results_dir "/home/ma-user/code/fenglang/Spoofing Detect/results" \
  --use_enhanced_labels \
  --label_type "conservative_spoofing"

# 跳过基线，只运行模型优化，指定结果目录
python run_optimization_pipeline.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --results_dir "/home/ma-user/code/fenglang/Spoofing Detect/results" \
  --skip_baseline \
  --experiments enhanced ensemble

# 快速测试
python run_optimization_pipeline.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --results_dir "/home/ma-user/code/fenglang/Spoofing Detect/results" \
  --skip_baseline \
  --experiments enhanced
""" 