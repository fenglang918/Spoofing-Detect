#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spoofing Detection Pipeline with Extended Labels
-----------------------------------------------
基于多种虚假报单模式的完整检测流程：
• 定义多种虚假报单模式
• 生成扩展标签 (Extended Labels) 
• 训练评估各种模式的预测效果
• 提供完整的从数据到评估的pipeline
"""

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SpoofingPattern:
    """虚假报单模式定义"""
    
    def __init__(self, name: str, description: str, rules: dict):
        self.name = name
        self.description = description
        self.rules = rules
    
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """检测该模式的虚假报单"""
        raise NotImplementedError

class QuickCancelPattern(SpoofingPattern):
    """快速撤单模式 - 在最佳价位提交大单后快速撤单"""
    
    def __init__(self):
        super().__init__(
            name="quick_cancel",
            description="在最佳价位提交大单后快速撤单，影响市场流动性",
            rules={
                "survival_time_ms": 100,
                "at_best_price": True,
                "large_order_quantile": 0.8
            }
        )
    
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """检测快速撤单模式"""
        conditions = [
            df['存活时间_ms'] < self.rules['survival_time_ms'],
            df['事件类型'] == '撤单',
            ((df['委托价格'] == df['bid1']) & (df['方向_委托'] == '买')) |
            ((df['委托价格'] == df['ask1']) & (df['方向_委托'] == '卖')),
            df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', self.rules['large_order_quantile'])
        ]
        return np.all(conditions, axis=0).astype(int)

class PriceManipulationPattern(SpoofingPattern):
    """价格操纵模式 - 激进定价后快速撤单"""
    
    def __init__(self):
        super().__init__(
            name="price_manipulation", 
            description="通过激进定价后快速撤单来操纵价格",
            rules={
                "survival_time_ms": 500,
                "price_aggressiveness_threshold": 2.0,
                "large_order_quantile": 0.75
            }
        )
    
    def detect(self, df: pd.DataFrame) -> pd.Series:
        conditions = [
            df['存活时间_ms'] < self.rules['survival_time_ms'],
            df['事件类型'] == '撤单',
            df['price_aggressiveness'].abs() > self.rules['price_aggressiveness_threshold'],
            df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', self.rules['large_order_quantile'])
        ]
        return np.all(conditions, axis=0).astype(int)

class FakeLiquidityPattern(SpoofingPattern):
    """虚假流动性模式 - 在最佳价位提供虚假流动性"""
    
    def __init__(self):
        super().__init__(
            name="fake_liquidity",
            description="在最佳价位提供虚假流动性后快速撤单",
            rules={
                "survival_time_ms": 200,
                "at_touch": True,
                "large_order_quantile": 0.9
            }
        )
    
    def detect(self, df: pd.DataFrame) -> pd.Series:
        conditions = [
            df['存活时间_ms'] < self.rules['survival_time_ms'],
            df['事件类型'] == '撤单',
            ((df['委托价格'] == df['bid1']) | (df['委托价格'] == df['ask1'])),
            df['委托数量'] > df.groupby('ticker')['委托数量'].transform('quantile', self.rules['large_order_quantile'])
        ]
        return np.all(conditions, axis=0).astype(int)

class LayeringPattern(SpoofingPattern):
    """分层下单模式 - 通过分层订单误导市场"""
    
    def __init__(self):
        super().__init__(
            name="layering",
            description="通过分层下单模式误导市场深度感知",
            rules={
                "survival_time_ms": 1000,
                "requires_layering_score": True
            }
        )
    
    def detect(self, df: pd.DataFrame) -> pd.Series:
        if 'layering_score' not in df.columns:
            return pd.Series(0, index=df.index)
        
        conditions = [
            df['layering_score'] > 0,
            df['存活时间_ms'] < self.rules['survival_time_ms'],
            df['事件类型'] == '撤单'
        ]
        return np.all(conditions, axis=0).astype(int)

class VolatilePeriodPattern(SpoofingPattern):
    """波动时段操纵模式 - 在市场活跃时段的异常行为"""
    
    def __init__(self):
        super().__init__(
            name="volatile_period",
            description="在开盘收盘等波动时段进行操纵",
            rules={
                "survival_time_ms": 50,
                "size_multiplier": 1.0
            }
        )
    
    def detect(self, df: pd.DataFrame) -> pd.Series:
        # 市场活跃时段
        market_open = ((df['委托_datetime'].dt.time >= pd.to_datetime('09:30').time()) & 
                      (df['委托_datetime'].dt.time <= pd.to_datetime('10:30').time()))
        market_close = ((df['委托_datetime'].dt.time >= pd.to_datetime('14:00').time()) & 
                       (df['委托_datetime'].dt.time <= pd.to_datetime('15:00').time()))
        
        conditions = [
            df['存活时间_ms'] < self.rules['survival_time_ms'],
            df['事件类型'] == '撤单',
            market_open | market_close,
            df['委托数量'] > df.groupby('ticker')['委托数量'].transform('median') * self.rules['size_multiplier']
        ]
        return np.all(conditions, axis=0).astype(int)

class SpoofingDetectionPipeline:
    """虚假报单检测Pipeline"""
    
    def __init__(self):
        self.patterns = [
            QuickCancelPattern(),
            PriceManipulationPattern(), 
            FakeLiquidityPattern(),
            LayeringPattern(),
            VolatilePeriodPattern()
        ]
        
        self.safe_features = [
            # 行情特征（委托时刻可观测）
            "bid1", "ask1", "prev_close", "mid_price", "spread",
            # 价格特征（委托时刻可观测）
            "delta_mid", "pct_spread", "price_dev_prevclose",
            # 订单特征（委托时刻可观测）
            "is_buy", "log_qty",
            # 历史统计特征（只使用过去的信息）
            "orders_100ms", "cancels_5s",
            # 时间特征
            "time_sin", "time_cos", "in_auction",
            # 增强特征（基于委托时刻的信息）
            "book_imbalance", "price_aggressiveness"
        ]
        
        self.leakage_features = [
            "final_survival_time_ms", "total_events", "total_traded_qty",
            "num_trades", "num_cancels", "is_fully_filled", "layering_score"
        ]
    
    def load_data(self, data_root: Path) -> pd.DataFrame:
        """加载和合并数据"""
        print("📥 Loading data...")
        
        # 加载特征
        feat_files = list((data_root / "features_select").glob("X_*.parquet"))
        df_features = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
        
        # 加载标签
        label_files = list((data_root / "labels_select").glob("labels_*.parquet"))
        df_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
        
        # 合并数据
        df = df_features.merge(df_labels, on=['自然日', 'ticker', '交易所委托号'], how='inner')
        
        print(f"  Features: {df_features.shape}")
        print(f"  Labels: {df_labels.shape}")
        print(f"  Merged: {df.shape}")
        
        return df
    
    def generate_extended_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成扩展标签"""
        print("\n🏷️ Generating Extended Labels...")
        
        # 确保必要的列存在
        if '委托_datetime' not in df.columns:
            df['委托_datetime'] = pd.to_datetime(df['自然日'].astype(str), format='%Y%m%d')
        else:
            df['委托_datetime'] = pd.to_datetime(df['委托_datetime'])
        
        # 为每种模式生成标签
        pattern_results = {}
        for pattern in self.patterns:
            try:
                df[pattern.name] = pattern.detect(df)
                pos_count = df[pattern.name].sum()
                pos_rate = pos_count / len(df) * 100
                pattern_results[pattern.name] = {
                    'count': pos_count,
                    'rate': pos_rate,
                    'description': pattern.description
                }
                print(f"  {pattern.name:<20}: {pos_count:>6,} ({pos_rate:>6.3f}%) - {pattern.description}")
            except Exception as e:
                print(f"  ⚠️ {pattern.name} failed: {e}")
                df[pattern.name] = 0
                pattern_results[pattern.name] = {'count': 0, 'rate': 0.0, 'description': pattern.description}
        
        # 组合标签
        pattern_cols = [p.name for p in self.patterns]
        df['extended_liberal'] = (df[pattern_cols].sum(axis=1) >= 1).astype(int)  # 任意模式
        df['extended_moderate'] = (df[pattern_cols].sum(axis=1) >= 2).astype(int)  # 两种模式
        df['extended_strict'] = (df[pattern_cols].sum(axis=1) >= 3).astype(int)    # 三种模式
        
        # 组合标签统计
        for label_type in ['extended_liberal', 'extended_moderate', 'extended_strict']:
            pos_count = df[label_type].sum()
            pos_rate = pos_count / len(df) * 100
            print(f"  {label_type:<20}: {pos_count:>6,} ({pos_rate:>6.3f}%)")
        
        return df, pattern_results
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """准备训练特征，移除数据泄露"""
        print("\n🔧 Preparing features...")
        
        # 移除泄露特征
        df_clean = df.drop(columns=self.leakage_features, errors='ignore')
        
        # 获取可用的安全特征
        available_features = [f for f in self.safe_features if f in df_clean.columns]
        
        print(f"  Safe features: {len(available_features)}")
        print(f"  Removed leakage features: {len([f for f in self.leakage_features if f in df.columns])}")
        
        return df_clean, available_features
    
    def train_evaluate_model(self, X_train, y_train, X_valid, y_valid, label_name: str) -> Dict:
        """训练和评估单个模型"""
        pos_count = y_train.sum()
        if pos_count == 0:
            return None
        
        # 数据平衡
        pos_indices = y_train[y_train == 1].index
        neg_indices = y_train[y_train == 0].index
        
        balance_ratio = 20
        target_neg_size = min(len(pos_indices) * balance_ratio, len(neg_indices))
        selected_neg_indices = np.random.choice(neg_indices, target_neg_size, replace=False)
        
        selected_indices = np.concatenate([pos_indices, selected_neg_indices])
        X_train_balanced = X_train.loc[selected_indices]
        y_train_balanced = y_train.loc[selected_indices]
        
        # 训练模型
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='average_precision',
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=10,
            reg_lambda=10,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        # 预测和评估
        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        
        metrics = {
            'PR-AUC': average_precision_score(y_valid, y_pred_proba),
            'ROC-AUC': roc_auc_score(y_valid, y_pred_proba),
            'positive_samples': pos_count,
            'positive_rate': pos_count / len(y_train) * 100
        }
        
        # Precision at K
        for k in [0.001, 0.005, 0.01, 0.05]:
            k_int = max(1, int(len(y_valid) * k))
            top_k_idx = y_pred_proba.argsort()[::-1][:k_int]
            prec_k = y_valid.iloc[top_k_idx].mean()
            metrics[f'Precision@{k*100:.1f}%'] = prec_k
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred_proba
        }
    
    def run_full_pipeline(self, data_root: str, train_regex: str, valid_regex: str, results_dir: str = None):
        """运行完整pipeline"""
        print("🚀 Spoofing Detection Pipeline with Extended Labels")
        print("=" * 80)
        
        data_root = Path(data_root)
        if results_dir:
            results_dir = Path(results_dir)
        else:
            results_dir = data_root / "extended_labels_results"
        results_dir.mkdir(exist_ok=True)
        
        # 1. 加载数据
        df = self.load_data(data_root)
        
        # 2. 生成扩展标签
        df, pattern_results = self.generate_extended_labels(df)
        
        # 3. 准备特征
        df_clean, feature_cols = self.prepare_features(df)
        
        # 4. 数据切分
        train_mask = df_clean["自然日"].astype(str).str.contains(train_regex)
        valid_mask = df_clean["自然日"].astype(str).str.contains(valid_regex)
        
        df_train = df_clean[train_mask].copy()
        df_valid = df_clean[valid_mask].copy()
        
        print(f"\n📅 Data split:")
        print(f"  Training: {len(df_train):,} samples")
        print(f"  Validation: {len(df_valid):,} samples")
        print(f"  Features: {len(feature_cols)}")
        
        X_train = df_train[feature_cols].fillna(0)
        X_valid = df_valid[feature_cols].fillna(0)
        
        # 5. 训练评估所有标签策略
        print(f"\n🎯 Training and Evaluation:")
        
        all_results = {}
        
        # 原始标签
        if 'y_label' in df_train.columns:
            result = self.train_evaluate_model(
                X_train, df_train['y_label'], X_valid, df_valid['y_label'], 'Original'
            )
            if result:
                all_results['original'] = result['metrics']
                print(f"  Original Labels    : PR-AUC={result['metrics']['PR-AUC']:.6f}")
        
        # 各种虚假报单模式
        for pattern in self.patterns:
            if pattern.name in df_train.columns:
                result = self.train_evaluate_model(
                    X_train, df_train[pattern.name], X_valid, df_valid[pattern.name], pattern.name
                )
                if result:
                    all_results[pattern.name] = result['metrics']
                    print(f"  {pattern.name:<15}: PR-AUC={result['metrics']['PR-AUC']:.6f}")
        
        # 组合扩展标签
        for ext_label in ['extended_liberal', 'extended_moderate', 'extended_strict']:
            if ext_label in df_train.columns:
                result = self.train_evaluate_model(
                    X_train, df_train[ext_label], X_valid, df_valid[ext_label], ext_label
                )
                if result:
                    all_results[ext_label] = result['metrics']
                    print(f"  {ext_label:<15}: PR-AUC={result['metrics']['PR-AUC']:.6f}")
        
        # 6. 生成详细报告
        self.generate_report(all_results, pattern_results, results_dir)
        
        print(f"\n💾 Results saved to: {results_dir}")
        return all_results
    
    def generate_report(self, all_results: Dict, pattern_results: Dict, results_dir: Path):
        """生成详细的对比报告"""
        print(f"\n📊 Extended Labels Performance Report")
        print("=" * 80)
        
        # 创建对比表格
        comparison_df = pd.DataFrame(all_results).T
        
        # 主要指标对比
        key_metrics = ['PR-AUC', 'ROC-AUC', 'Precision@0.1%', 'Precision@1.0%']
        available_metrics = [m for m in key_metrics if m in comparison_df.columns]
        
        print(f"\n🎯 Performance Comparison:")
        print(comparison_df[available_metrics].round(6))
        
        # 最佳策略
        if 'PR-AUC' in comparison_df.columns:
            best_strategy = comparison_df['PR-AUC'].idxmax()
            best_score = comparison_df.loc[best_strategy, 'PR-AUC']
            print(f"\n🏆 Best Strategy: {best_strategy} (PR-AUC: {best_score:.6f})")
        
        # 模式分析
        print(f"\n🔍 Spoofing Pattern Analysis:")
        for pattern_name, stats in pattern_results.items():
            print(f"  {pattern_name:<20}: {stats['count']:>6,} samples ({stats['rate']:>6.3f}%)")
            print(f"  {'':>22}  {stats['description']}")
        
        # 保存结果
        comparison_df.to_csv(results_dir / "extended_labels_comparison.csv", float_format='%.8f')
        
        with open(results_dir / "pattern_analysis.json", 'w') as f:
            json.dump(pattern_results, f, indent=2)
        
        with open(results_dir / "performance_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

def main():
    parser = argparse.ArgumentParser(description="Spoofing Detection Pipeline with Extended Labels")
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--train_regex", default="202503|202504", help="训练数据日期正则")
    parser.add_argument("--valid_regex", default="202505", help="验证数据日期正则")
    parser.add_argument("--results_dir", help="结果保存目录")
    
    args = parser.parse_args()
    
    # 运行pipeline
    pipeline = SpoofingDetectionPipeline()
    results = pipeline.run_full_pipeline(
        data_root=args.data_root,
        train_regex=args.train_regex,
        valid_regex=args.valid_regex,
        results_dir=args.results_dir
    )

if __name__ == "__main__":
    main()

"""
使用示例：

# 运行完整的虚假报单检测pipeline
python scripts/spoofing_detection_pipeline.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505"

# 指定结果保存目录
python scripts/spoofing_detection_pipeline.py \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data" \
  --results_dir "./extended_labels_analysis"
""" 