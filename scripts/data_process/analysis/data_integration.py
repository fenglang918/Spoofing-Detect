#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_integration.py
────────────────────────────────────────
数据整合分析模块 - 第三阶段数据整合与评估
• 整合特征和标签数据
• 全局数据质量分析
• 特征质量评估和筛选
• 生成训练就绪数据
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import argparse
import json
import warnings
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

console = Console()
warnings.filterwarnings('ignore')

class DataIntegrator:
    """数据整合器"""
    
    def __init__(self, target_col: str = "y_label"):
        """
        初始化数据整合器
        
        Args:
            target_col: 目标标签列名
        """
        self.target_col = target_col
        self.console = Console()
        
        # 定义信息泄露特征模式
        self.leakage_patterns = [
            r'total_.*',           # 总量统计
            r'num_.*',            # 数量统计  
            r'final_.*',          # 最终状态
            r'.*_survival.*',     # 生存时间相关
            r'is_fully_filled',   # 成交状态
            r'flag_R[12]',        # 中间标签变量
            r'.*_cancel$',        # 撤单相关（但保留is_cancel_event）
            r'存活时间_ms',       # 原始存活时间
        ]
        
        # 定义安全的核心特征（不应被移除）
        self.safe_core_features = {
            # 主键字段
            '自然日', 'ticker', '交易所委托号',
            # 盘口快照特征
            'bid1', 'ask1', 'mid_price', 'spread', 'prev_close', 
            'bid_vol1', 'ask_vol1', 'pct_spread',
            # 订单基础特征
            'log_qty', 'is_buy', 'delta_mid', 'price_dev_prevclose_bps',
            # 短期历史窗口特征
            'orders_100ms', 'orders_1s', 'cancels_100ms', 'cancels_1s', 'cancels_5s',
            'cancel_ratio_100ms', 'cancel_ratio_1s', 'trades_1s',
            # 时间和市场状态特征
            'time_sin', 'time_cos', 'in_auction',
            # 订单簿压力特征
            'book_imbalance', 'price_aggressiveness', 'cluster_score',
            # 事件类型
            'is_cancel_event'
        }
    
    def load_and_integrate_data(self, 
                               features_dir: Path, 
                               labels_dir: Path,
                               feature_pattern: str = "X_*.parquet",
                               label_pattern: str = "labels_*.parquet",
                               dates: Optional[List[str]] = None) -> pd.DataFrame:
        """
        加载并整合特征和标签数据
        
        Args:
            features_dir: 特征数据目录
            labels_dir: 标签数据目录
            feature_pattern: 特征文件模式
            label_pattern: 标签文件模式
            dates: 指定处理的日期列表
            
        Returns:
            整合后的DataFrame
        """
        self.console.print(f"[bold green]📂 加载并整合数据...[/bold green]")
        
        # 查找特征文件
        feat_files = list(features_dir.glob(feature_pattern))
        if not feat_files:
            raise FileNotFoundError(f"未找到特征文件: {features_dir}/{feature_pattern}")
        
        # 查找标签文件
        label_files = list(labels_dir.glob(label_pattern))
        if not label_files:
            raise FileNotFoundError(f"未找到标签文件: {labels_dir}/{label_pattern}")
        
        # 日期筛选
        if dates:
            target_dates = set(dates)
            feat_files = [f for f in feat_files if any(d in f.name for d in target_dates)]
            label_files = [f for f in label_files if any(d in f.name for d in target_dates)]
        
        self.console.print(f"找到 {len(feat_files)} 个特征文件, {len(label_files)} 个标签文件")
        
        # 加载特征数据
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            # 加载特征
            feat_task = progress.add_task("加载特征数据...", total=len(feat_files))
            feat_dfs = []
            for f in feat_files:
                try:
                    df = pd.read_parquet(f)
                    feat_dfs.append(df)
                    progress.advance(feat_task)
                except Exception as e:
                    self.console.print(f"[yellow]警告: 加载特征文件失败 {f}: {e}[/yellow]")
                    progress.advance(feat_task)
            
            # 加载标签
            label_task = progress.add_task("加载标签数据...", total=len(label_files))
            label_dfs = []
            for f in label_files:
                try:
                    df = pd.read_parquet(f)
                    label_dfs.append(df)
                    progress.advance(label_task)
                except Exception as e:
                    self.console.print(f"[yellow]警告: 加载标签文件失败 {f}: {e}[/yellow]")
                    progress.advance(label_task)
            
            # 合并数据
            merge_task = progress.add_task("合并数据...", total=100)
            
            if not feat_dfs or not label_dfs:
                raise ValueError("无有效的特征或标签数据")
            
            df_features = pd.concat(feat_dfs, ignore_index=True)
            df_labels = pd.concat(label_dfs, ignore_index=True)
            
            progress.update(merge_task, advance=50)
            
            # 合并特征和标签
            df_integrated = df_features.merge(
                df_labels, 
                on=['自然日', 'ticker', '交易所委托号'], 
                how='inner'
            )
            
            progress.update(merge_task, advance=50)
        
        self.console.print(f"[green]✅ 整合完成: {df_integrated.shape}[/green]")
        return df_integrated
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        全面分析数据质量
        
        Args:
            df: 整合后的数据
            
        Returns:
            分析结果字典
        """
        self.console.print(f"\n[bold cyan]🔍 数据质量分析[/bold cyan]")
        
        results = {}
        
        # 1. 基础统计
        results['basic_stats'] = self._get_basic_stats(df)
        
        # 2. 识别问题特征
        results['problematic_features'] = self._identify_problematic_features(df)
        
        # 3. 信息泄露检测
        results['leakage_features'] = self._detect_information_leakage(df)
        
        # 4. 特征重要性分析
        if self.target_col in df.columns:
            results['feature_importance'] = self._analyze_feature_importance(df)
        
        # 5. 数据分布分析
        results['distribution_analysis'] = self._analyze_distributions(df)
        
        # 6. 生成过滤建议
        results['filter_recommendations'] = self._generate_filter_recommendations(results)
        
        return results
    
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict:
        """获取基础统计信息"""
        feature_cols = [col for col in df.columns 
                       if col not in ['自然日', 'ticker', '交易所委托号', self.target_col]]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {
            'total_features': len(feature_cols),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(feature_cols) - len(numeric_cols),
            'total_samples': len(df),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        
        # 按日期统计
        if '自然日' in df.columns:
            stats['date_distribution'] = df['自然日'].value_counts().head(10).to_dict()
            stats['unique_dates'] = df['自然日'].nunique()
        
        # 按股票统计
        if 'ticker' in df.columns:
            stats['ticker_distribution'] = df['ticker'].value_counts().head(10).to_dict()
            stats['unique_tickers'] = df['ticker'].nunique()
        
        # 标签分布
        if self.target_col in df.columns:
            stats['label_distribution'] = df[self.target_col].value_counts().to_dict()
            stats['positive_rate'] = df[self.target_col].mean()
        
        return stats
    
    def _identify_problematic_features(self, df: pd.DataFrame) -> Dict:
        """识别问题特征"""
        feature_cols = [col for col in df.columns 
                       if col not in ['自然日', 'ticker', '交易所委托号', self.target_col]]
        
        problems = {
            'constant_features': [],      # 常数列
            'low_variance_features': [],  # 低方差特征
            'high_missing_features': [],  # 高缺失率特征
            'duplicate_features': [],     # 重复特征
            'inf_nan_features': [],       # 包含无穷值的特征
            'outlier_features': [],       # 异常值过多的特征
        }
        
        for col in feature_cols:
            # 常数列检测
            if df[col].nunique() <= 1:
                problems['constant_features'].append(col)
                continue
            
            # 高缺失率检测
            missing_rate = df[col].isnull().mean()
            if missing_rate > 0.9:
                problems['high_missing_features'].append(col)
                continue
            
            # 数值特征的进一步检测
            if df[col].dtype in ['int64', 'float64']:
                # 低方差检测
                try:
                    if df[col].var() < 1e-10:
                        problems['low_variance_features'].append(col)
                except:
                    pass
                
                # 无穷值检测
                if np.isinf(df[col]).any():
                    problems['inf_nan_features'].append(col)
                
                # 异常值检测（IQR方法）
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).mean()
                    if outliers > 0.1:  # 超过10%的异常值
                        problems['outlier_features'].append((col, outliers))
                except:
                    pass
        
        # 重复特征检测
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                duplicate_pairs = []
                for col in upper_triangle.columns:
                    high_corr_features = upper_triangle.index[upper_triangle[col] > 0.99].tolist()
                    for feat in high_corr_features:
                        duplicate_pairs.append((feat, col))
                
                # 只保留一个，移除其他
                to_remove = set()
                for feat1, feat2 in duplicate_pairs:
                    to_remove.add(feat2)
                
                problems['duplicate_features'] = list(to_remove)
            except:
                pass
        
        return problems
    
    def _detect_information_leakage(self, df: pd.DataFrame) -> Dict:
        """检测信息泄露特征"""
        import re
        
        leakage_info = {
            'pattern_matches': {},
            'suspicious_features': [],
            'high_correlation_features': {}
        }
        
        feature_cols = [col for col in df.columns 
                       if col not in ['自然日', 'ticker', '交易所委托号']]
        
        # 1. 模式匹配检测
        for pattern in self.leakage_patterns:
            matches = [col for col in feature_cols if re.match(pattern, col)]
            if matches:
                leakage_info['pattern_matches'][pattern] = matches
        
        # 2. 与目标变量的异常相关性检测
        if self.target_col in df.columns:
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != self.target_col:
                    try:
                        corr = df[col].corr(df[self.target_col])
                        if abs(corr) > 0.8:  # 异常高的相关性
                            leakage_info['high_correlation_features'][col] = corr
                    except:
                        pass
        
        # 3. 已知的信息泄露特征
        known_leakage = [
            'total_events', 'total_traded_qty', 'num_trades', 'num_cancels',
            'final_survival_time_ms', 'is_fully_filled', 'survival_time_ms',
            'final_state', 'is_cancel', '存活时间_ms'
        ]
        
        leakage_info['suspicious_features'] = [
            col for col in feature_cols if col in known_leakage
        ]
        
        return leakage_info
    
    def _analyze_feature_importance(self, df: pd.DataFrame) -> Dict:
        """分析特征重要性"""
        feature_cols = [col for col in df.columns 
                       if col not in ['自然日', 'ticker', '交易所委托号', self.target_col]]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        importance_info = {
            'correlations': {},
            'mutual_information': {},
            'statistical_tests': {}
        }
        
        if len(numeric_cols) == 0:
            return importance_info
        
        # 1. 相关性分析
        for col in numeric_cols:
            try:
                corr = df[col].corr(df[self.target_col])
                if not np.isnan(corr):
                    importance_info['correlations'][col] = abs(corr)
            except:
                pass
        
        # 2. 统计检验（卡方检验、t检验等）
        try:
            from scipy.stats import chi2_contingency, ttest_ind
            
            for col in numeric_cols[:20]:  # 限制数量避免计算过久
                try:
                    # 对于连续变量，进行t检验
                    group1 = df[df[self.target_col] == 1][col].dropna()
                    group0 = df[df[self.target_col] == 0][col].dropna()
                    
                    if len(group1) > 10 and len(group0) > 10:
                        _, p_value = ttest_ind(group1, group0)
                        importance_info['statistical_tests'][col] = 1 - p_value  # 转换为重要性分数
                except:
                    pass
        except ImportError:
            pass
        
        return importance_info
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict:
        """分析数据分布"""
        distribution_info = {
            'skewness': {},
            'kurtosis': {},
            'zero_ratios': {},
            'unique_ratios': {}
        }
        
        feature_cols = [col for col in df.columns 
                       if col not in ['自然日', 'ticker', '交易所委托号', self.target_col]]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                # 偏度
                distribution_info['skewness'][col] = df[col].skew()
                
                # 峰度
                distribution_info['kurtosis'][col] = df[col].kurtosis()
                
                # 零值比例
                distribution_info['zero_ratios'][col] = (df[col] == 0).mean()
                
                # 唯一值比例
                distribution_info['unique_ratios'][col] = df[col].nunique() / len(df)
                
            except:
                pass
        
        return distribution_info
    
    def _generate_filter_recommendations(self, analysis_results: Dict) -> Dict:
        """生成过滤建议"""
        recommendations = {
            'features_to_remove': set(),
            'features_to_investigate': set(),
            'safe_features': set(),
            'priority_features': set(),
            'reasons': {}
        }
        
        # 1. 必须移除的特征
        problems = analysis_results.get('problematic_features', {})
        
        for feat in problems.get('constant_features', []):
            recommendations['features_to_remove'].add(feat)
            recommendations['reasons'][feat] = '常数列'
        
        for feat in problems.get('low_variance_features', []):
            recommendations['features_to_remove'].add(feat)
            recommendations['reasons'][feat] = '低方差'
        
        for feat in problems.get('high_missing_features', []):
            recommendations['features_to_remove'].add(feat)
            recommendations['reasons'][feat] = '高缺失率'
        
        for feat in problems.get('duplicate_features', []):
            recommendations['features_to_remove'].add(feat)
            recommendations['reasons'][feat] = '重复特征'
        
        # 2. 信息泄露特征
        leakage = analysis_results.get('leakage_features', {})
        
        for pattern, matches in leakage.get('pattern_matches', {}).items():
            for feat in matches:
                if feat not in self.safe_core_features:
                    recommendations['features_to_remove'].add(feat)
                    recommendations['reasons'][feat] = f'疑似泄露(模式:{pattern})'
        
        for feat in leakage.get('suspicious_features', []):
            if feat not in self.safe_core_features:
                recommendations['features_to_remove'].add(feat)
                recommendations['reasons'][feat] = '已知信息泄露特征'
        
        for feat, corr in leakage.get('high_correlation_features', {}).items():
            if feat not in self.safe_core_features:
                recommendations['features_to_investigate'].add(feat)
                recommendations['reasons'][feat] = f'与目标异常相关({corr:.3f})'
        
        # 3. 安全特征
        recommendations['safe_features'] = self.safe_core_features.copy()
        
        # 4. 优先特征（基于重要性）
        importance = analysis_results.get('feature_importance', {})
        correlations = importance.get('correlations', {})
        
        if correlations:
            sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            top_n = min(20, max(5, len(sorted_corr) // 5))
            for feat, _ in sorted_corr[:top_n]:
                if feat not in recommendations['features_to_remove']:
                    recommendations['priority_features'].add(feat)
        
        return recommendations
    
    def apply_filters(self, df: pd.DataFrame, filter_recommendations: Dict) -> pd.DataFrame:
        """应用过滤建议"""
        features_to_remove = filter_recommendations.get('features_to_remove', set())
        
        # 确保不移除主键和目标列
        protected_cols = {'自然日', 'ticker', '交易所委托号', self.target_col}
        features_to_remove = features_to_remove - protected_cols
        
        if features_to_remove:
            self.console.print(f"[yellow]移除 {len(features_to_remove)} 个特征[/yellow]")
            df_filtered = df.drop(columns=list(features_to_remove), errors='ignore')
        else:
            df_filtered = df.copy()
        
        return df_filtered
    
    def print_analysis_report(self, analysis_results: Dict):
        """打印分析报告"""
        self.console.print(f"\n[bold green]📋 数据质量分析报告[/bold green]")
        self.console.print("=" * 80)
        
        # 1. 基础统计
        stats = analysis_results.get('basic_stats', {})
        table = Table(title="数据概览")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="magenta")
        
        table.add_row("总特征数", str(stats.get('total_features', 0)))
        table.add_row("数值特征", str(stats.get('numeric_features', 0)))
        table.add_row("总样本数", f"{stats.get('total_samples', 0):,}")
        table.add_row("内存使用", f"{stats.get('memory_usage_mb', 0):.1f} MB")
        table.add_row("日期数量", str(stats.get('unique_dates', 0)))
        table.add_row("股票数量", str(stats.get('unique_tickers', 0)))
        
        if 'positive_rate' in stats:
            table.add_row("正样本比例", f"{stats['positive_rate']*100:.4f}%")
        
        self.console.print(table)
        
        # 2. 问题特征
        problems = analysis_results.get('problematic_features', {})
        if any(problems.values()):
            self.console.print(f"\n[bold red]🚫 问题特征统计[/bold red]")
            for problem_type, features in problems.items():
                if features:
                    if problem_type == 'outlier_features':
                        self.console.print(f"  • {problem_type}: {len(features)} 个")
                    else:
                        self.console.print(f"  • {problem_type}: {len(features)} 个")
                        if len(features) <= 5:
                            self.console.print(f"    {features}")
                        else:
                            self.console.print(f"    {features[:5]}...")
        
        # 3. 信息泄露
        leakage = analysis_results.get('leakage_features', {})
        total_leakage = (len(leakage.get('suspicious_features', [])) + 
                        sum(len(matches) for matches in leakage.get('pattern_matches', {}).values()))
        
        self.console.print(f"\n[bold yellow]🔍 信息泄露检测[/bold yellow]")
        self.console.print(f"  • 疑似泄露特征: {total_leakage} 个")
        self.console.print(f"  • 异常高相关性: {len(leakage.get('high_correlation_features', {}))} 个")
        
        # 4. 过滤建议
        recommendations = analysis_results.get('filter_recommendations', {})
        if recommendations:
            self.console.print(f"\n[bold cyan]💡 过滤建议[/bold cyan]")
            self.console.print(f"  • 建议移除: {len(recommendations.get('features_to_remove', set()))} 个特征")
            self.console.print(f"  • 需要调查: {len(recommendations.get('features_to_investigate', set()))} 个特征")
            self.console.print(f"  • 优先特征: {len(recommendations.get('priority_features', set()))} 个特征")
            self.console.print(f"  • 安全特征: {len(recommendations.get('safe_features', set()))} 个特征")

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="数据整合分析模块")
    parser.add_argument("--features_dir", required=True, help="特征数据目录")
    parser.add_argument("--labels_dir", required=True, help="标签数据目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--target_col", default="y_label", help="目标标签列名")
    parser.add_argument("--dates", nargs="*", help="指定处理日期")
    parser.add_argument("--apply_filter", action="store_true", help="应用过滤建议")
    
    args = parser.parse_args()
    
    try:
        # 创建整合器
        integrator = DataIntegrator(target_col=args.target_col)
        
        # 加载并整合数据
        df_integrated = integrator.load_and_integrate_data(
            features_dir=Path(args.features_dir),
            labels_dir=Path(args.labels_dir),
            dates=args.dates
        )
        
        # 分析数据质量
        analysis_results = integrator.analyze_data_quality(df_integrated)
        
        # 打印报告
        integrator.print_analysis_report(analysis_results)
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存分析结果
        with open(output_dir / "data_analysis.json", "w", encoding="utf-8") as f:
            # 转换set为list以便JSON序列化
            json_results = {}
            for key, value in analysis_results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, set):
                            json_results[key][k] = list(v)
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        # 保存原始整合数据
        df_integrated.to_parquet(output_dir / "integrated_data.parquet", index=False)
        console.print(f"[green]💾 原始整合数据已保存: {output_dir}/integrated_data.parquet[/green]")
        
        # 应用过滤（如果指定）
        if args.apply_filter:
            df_filtered = integrator.apply_filters(
                df_integrated, 
                analysis_results.get('filter_recommendations', {})
            )
            
            df_filtered.to_parquet(output_dir / "filtered_data.parquet", index=False)
            console.print(f"[green]💾 过滤后数据已保存: {output_dir}/filtered_data.parquet[/green]")
            console.print(f"数据形状变化: {df_integrated.shape} → {df_filtered.shape}")
        
        console.print(f"[green]📄 分析结果已保存: {output_dir}/data_analysis.json[/green]")
        console.print(f"[bold green]✅ 数据整合分析完成！[/bold green]")
        
    except Exception as e:
        console.print(f"[red]❌ 错误: {e}[/red]")
        raise

if __name__ == "__main__":
    main()

"""
使用示例:

# 基础整合分析
python scripts/data_process/analysis/data_integration.py \
    --features_dir "/path/to/features" \
    --labels_dir "/path/to/labels" \
    --output_dir "/path/to/analysis_results"

# 整合分析并应用过滤
python scripts/data_process/analysis/data_integration.py \
    --features_dir "/path/to/features" \
    --labels_dir "/path/to/labels" \
    --output_dir "/path/to/analysis_results" \
    --apply_filter

# 指定日期和目标列
python scripts/data_process/analysis/data_integration.py \
    --features_dir "/path/to/features" \
    --labels_dir "/path/to/labels" \
    --output_dir "/path/to/analysis_results" \
    --dates 20230301 20230302 \
    --target_col "spoofing_label" \
    --apply_filter
""" 