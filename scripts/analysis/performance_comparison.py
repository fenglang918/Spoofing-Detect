#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Comparison and Analysis
-----------------------------------
对比不同模型配置的性能：
• 基线模型 vs 增强模型
• 不同采样策略的效果
• 特征重要性分析
• 模型诊断和建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

def load_results(results_dir):
    """加载不同实验的结果"""
    results = {}
    
    results_path = Path(results_dir)
    
    # 寻找结果文件
    for result_file in results_path.glob("*.json"):
        experiment_name = result_file.stem
        with open(result_file, 'r') as f:
            results[experiment_name] = json.load(f)
    
    return results

def create_performance_comparison(results):
    """创建性能对比图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 准备数据
    experiments = list(results.keys())
    metrics = ['PR-AUC', 'ROC-AUC', 'Precision@0.1%', 'Precision@0.5%', 'Precision@1.0%']
    
    # 1. 主要指标对比
    metric_data = []
    for exp in experiments:
        for metric in metrics:
            if metric in results[exp]:
                metric_data.append({
                    'Experiment': exp,
                    'Metric': metric,
                    'Value': results[exp][metric]
                })
    
    metric_df = pd.DataFrame(metric_data)
    
    # PR-AUC对比
    pr_auc_data = metric_df[metric_df['Metric'] == 'PR-AUC']
    axes[0, 0].bar(pr_auc_data['Experiment'], pr_auc_data['Value'])
    axes[0, 0].set_title('PR-AUC Comparison')
    axes[0, 0].set_ylabel('PR-AUC')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # ROC-AUC对比
    roc_auc_data = metric_df[metric_df['Metric'] == 'ROC-AUC']
    axes[0, 1].bar(roc_auc_data['Experiment'], roc_auc_data['Value'])
    axes[0, 1].set_title('ROC-AUC Comparison')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision@K对比
    precision_data = metric_df[metric_df['Metric'].str.contains('Precision@')]
    if not precision_data.empty:
        pivot_data = precision_data.pivot(index='Experiment', columns='Metric', values='Value')
        pivot_data.plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Precision@K Comparison')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. 训练时间对比
    if any('training_time' in results[exp] for exp in experiments):
        training_times = [results[exp].get('training_time', 0) for exp in experiments]
        axes[1, 0].bar(experiments, training_times)
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 3. 模型复杂度对比
    if any('n_features' in results[exp] for exp in experiments):
        n_features = [results[exp].get('n_features', 0) for exp in experiments]
        axes[1, 1].bar(experiments, n_features)
        axes[1, 1].set_title('Feature Count Comparison')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 4. 数据分布对比
    if any('positive_ratio' in results[exp] for exp in experiments):
        pos_ratios = [results[exp].get('positive_ratio', 0) * 100 for exp in experiments]
        axes[1, 2].bar(experiments, pos_ratios)
        axes[1, 2].set_title('Positive Sample Ratio')
        axes[1, 2].set_ylabel('Positive Ratio (%)')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def analyze_feature_importance(results):
    """分析特征重要性变化"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 收集所有实验的特征重要性
    all_features = set()
    for exp in results:
        if 'feature_importance' in results[exp]:
            all_features.update(results[exp]['feature_importance'].keys())
    
    if not all_features:
        print("No feature importance data found")
        return None
    
    # 创建特征重要性矩阵
    importance_matrix = []
    experiment_names = []
    
    for exp in results:
        if 'feature_importance' in results[exp]:
            importance_row = []
            for feature in sorted(all_features):
                importance_row.append(results[exp]['feature_importance'].get(feature, 0))
            importance_matrix.append(importance_row)
            experiment_names.append(exp)
    
    if importance_matrix:
        # 热力图
        importance_df = pd.DataFrame(
            importance_matrix, 
            columns=sorted(all_features),
            index=experiment_names
        )
        
        # 选择top特征
        top_features = importance_df.mean().nlargest(20).index
        
        sns.heatmap(
            importance_df[top_features], 
            annot=True, 
            cmap='viridis', 
            ax=axes[0]
        )
        axes[0].set_title('Feature Importance Heatmap (Top 20)')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Experiments')
        
        # 特征重要性变化
        if len(experiment_names) > 1:
            for feature in top_features[:5]:  # 只显示前5个特征
                values = importance_df[feature].values
                axes[1].plot(range(len(experiment_names)), values, marker='o', label=feature)
            
            axes[1].set_title('Top 5 Feature Importance Trends')
            axes[1].set_xlabel('Experiments')
            axes[1].set_ylabel('Importance')
            axes[1].set_xticks(range(len(experiment_names)))
            axes[1].set_xticklabels(experiment_names, rotation=45)
            axes[1].legend()
    
    plt.tight_layout()
    return fig

def generate_recommendations(results):
    """生成优化建议"""
    recommendations = []
    
    # 找到最佳性能的实验
    best_pr_auc = 0
    best_experiment = None
    
    for exp, result in results.items():
        pr_auc = result.get('PR-AUC')
        if pr_auc is not None and pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_experiment = exp
    
    if best_experiment:
        recommendations.append(f"🏆 Best performing model: {best_experiment} (PR-AUC: {best_pr_auc:.4f})")
    
    # 分析性能差异 - 修复除零错误
    pr_aucs = []
    for result in results.values():
        pr_auc = result.get('PR-AUC')
        if pr_auc is not None and pr_auc > 0:  # 确保非None且非零
            pr_aucs.append(pr_auc)
    
    if len(pr_aucs) > 1:
        min_pr_auc = min(pr_aucs)
        max_pr_auc = max(pr_aucs)
        
        if min_pr_auc > 0:  # 确保分母不为零
            improvement = (max_pr_auc - min_pr_auc) / min_pr_auc * 100
            if improvement > 50:
                recommendations.append(f"💡 Significant improvement achieved: {improvement:.1f}% gain from optimization")
            elif improvement < 10:
                recommendations.append("⚠️ Limited improvement - consider more advanced techniques")
        else:
            recommendations.append("⚠️ Some experiments had zero PR-AUC - check model training")
    elif len(pr_aucs) == 1:
        recommendations.append("ℹ️ Only one valid experiment result - need more experiments for comparison")
    else:
        recommendations.append("❌ No valid PR-AUC results found - check experiment outputs")
    
    # 检查数据不平衡问题
    for exp, result in results.items():
        if 'positive_ratio' in result and result['positive_ratio'] is not None:
            pos_ratio = result['positive_ratio'] * 100
            if pos_ratio < 0.1:
                recommendations.append(f"⚖️ Severe class imbalance in {exp} ({pos_ratio:.3f}%) - consider better sampling")
            elif pos_ratio > 10:
                recommendations.append(f"📈 Good class balance in {exp} ({pos_ratio:.1f}%)")
    
    # 特征数量分析
    feature_counts = []
    for result in results.values():
        n_features = result.get('n_features')
        if n_features is not None and n_features > 0:
            feature_counts.append(n_features)
    
    if len(feature_counts) > 1 and min(feature_counts) > 0:
        if max(feature_counts) > min(feature_counts) * 2:
            recommendations.append("🔧 Feature engineering made significant impact - continue exploring features")
    
    # 训练时间分析
    training_times = []
    for result in results.values():
        training_time = result.get('training_time')
        if training_time is not None and training_time > 0:
            training_times.append(training_time)
    
    if training_times and max(training_times) > 300:  # 5 minutes
        recommendations.append("⏱️ Long training time detected - consider model simplification for production")
    
    return recommendations

def create_summary_report(results, output_dir):
    """创建总结报告"""
    report = []
    report.append("# Spoofing Detection Model Performance Report\n")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 实验概览
    report.append("## Experiments Overview\n")
    for exp, result in results.items():
        report.append(f"### {exp}")
        
        # 安全格式化PR-AUC
        pr_auc = result.get('PR-AUC', 'N/A')
        if isinstance(pr_auc, (int, float)) and pr_auc is not None:
            report.append(f"- PR-AUC: {pr_auc:.6f}")
        else:
            report.append(f"- PR-AUC: {pr_auc}")
        
        # 安全格式化ROC-AUC
        roc_auc = result.get('ROC-AUC', 'N/A')
        if isinstance(roc_auc, (int, float)) and roc_auc is not None:
            report.append(f"- ROC-AUC: {roc_auc:.6f}")
        else:
            report.append(f"- ROC-AUC: {roc_auc}")
        
        # Features数量
        n_features = result.get('n_features', 'N/A')
        report.append(f"- Features: {n_features}")
        
        # 安全格式化训练时间
        training_time = result.get('training_time', 'N/A')
        if isinstance(training_time, (int, float)) and training_time is not None:
            report.append(f"- Training time: {training_time:.1f}s")
        else:
            report.append(f"- Training time: {training_time}")
        
        # 安全格式化正样本比例
        positive_ratio = result.get('positive_ratio', 'N/A')
        if isinstance(positive_ratio, (int, float)) and positive_ratio is not None:
            report.append(f"- Positive ratio: {positive_ratio:.4%}\n")
        else:
            report.append(f"- Positive ratio: {positive_ratio}\n")
    
    # 性能对比
    report.append("## Performance Comparison\n")
    metrics_df = pd.DataFrame(results).T
    if not metrics_df.empty:
        report.append(metrics_df.to_markdown())
        report.append("\n")
    
    # 建议
    recommendations = generate_recommendations(results)
    if recommendations:
        report.append("## Recommendations\n")
        for rec in recommendations:
            report.append(f"- {rec}")
        report.append("\n")
    
    # 保存报告
    output_path = Path(output_dir) / "performance_report.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"📊 Report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Results directory")
    parser.add_argument("--output_dir", default="analysis_output", help="Output directory")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载结果
    print("📁 Loading experiment results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("❌ No results found in the specified directory")
        return
    
    print(f"🔍 Found {len(results)} experiments: {list(results.keys())}")
    
    # 创建性能对比图
    print("📊 Creating performance comparison plots...")
    perf_fig = create_performance_comparison(results)
    perf_fig.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(perf_fig)
    
    # 特征重要性分析
    print("🔧 Analyzing feature importance...")
    feat_fig = analyze_feature_importance(results)
    if feat_fig:
        feat_fig.savefig(output_dir / "feature_importance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(feat_fig)
    
    # 生成建议
    print("💡 Generating recommendations...")
    recommendations = generate_recommendations(results)
    
    print("\n🎯 Key Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    # 创建总结报告
    print("📝 Creating summary report...")
    create_summary_report(results, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

"""
# 使用示例
python scripts/analysis/performance_comparison.py \
    --results_dir "experiments/results" \
    --output_dir "analysis_output"
""" 