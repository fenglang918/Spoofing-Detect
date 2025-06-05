#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Spoofing Detection Pipeline with Extended Labels
--------------------------------------------------------
完整的端到端虚假报单检测系统：
• Step 1: 原始数据合并 (merge_order_trade)
• Step 2: 特征工程和扩展标签生成 (run_etl_from_event) 
• Step 3: 多种虚假报单模式训练和评估
• 提供统一的配置和架构
"""

import argparse
import subprocess
import sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SpoofingPattern:
    """虚假报单模式定义"""
    
    def __init__(self, name: str, description: str, rules: dict):
        self.name = name
        self.description = description
        self.rules = rules
    
    def generate_label_logic(self) -> str:
        """生成标签逻辑的代码字符串"""
        raise NotImplementedError

class QuickCancelImpactPattern(SpoofingPattern):
    """快速撤单冲击模式"""
    
    def __init__(self):
        super().__init__(
            name="quick_cancel_impact",
            description="在最佳价位的大单快速撤单",
            rules={
                "survival_time_ms": 100,
                "event_type": "撤单",
                "at_best_price": True,
                "large_order_multiplier": 2.0
            }
        )
    
    def generate_label_logic(self) -> str:
        return f"""
# 规则1: 快速撤单冲击 - 在最佳价位的大单快速撤单
conditions_r1 = [
    df_pd['存活时间_ms'] < {self.rules['survival_time_ms']},
    df_pd['事件类型'] == '{self.rules['event_type']}',
    (df_pd['at_bid'] == 1) | (df_pd['at_ask'] == 1),
    df_pd['委托数量'] > df_pd.groupby('ticker')['委托数量'].transform('median') * {self.rules['large_order_multiplier']}
]
"""

class PriceManipulationPattern(SpoofingPattern):
    """价格操纵模式"""
    
    def __init__(self):
        super().__init__(
            name="price_manipulation",
            description="激进定价但快速撤单",
            rules={
                "survival_time_ms": 500,
                "event_type": "撤单", 
                "price_aggressiveness_threshold": 2.0,
                "large_order_quantile": 0.75
            }
        )
    
    def generate_label_logic(self) -> str:
        return f"""
# 规则2: 价格操纵 - 激进定价但快速撤单
conditions_r2 = [
    df_pd['存活时间_ms'] < {self.rules['survival_time_ms']},
    df_pd['事件类型'] == '{self.rules['event_type']}',
    np.abs(df_pd['price_aggressiveness']) > {self.rules['price_aggressiveness_threshold']},
    df_pd['委托数量'] > df_pd.groupby('ticker')['委托数量'].transform('quantile', {self.rules['large_order_quantile']})
]
"""

class FakeLiquidityPattern(SpoofingPattern):
    """虚假流动性模式"""
    
    def __init__(self):
        super().__init__(
            name="fake_liquidity",
            description="最佳价位大单快速撤单",
            rules={
                "survival_time_ms": 200,
                "event_type": "撤单",
                "at_touch": True,
                "large_order_quantile": 0.9
            }
        )
    
    def generate_label_logic(self) -> str:
        return f"""
# 规则3: 虚假流动性 - 最佳价位大单快速撤单
conditions_r3 = [
    df_pd['存活时间_ms'] < {self.rules['survival_time_ms']},
    df_pd['事件类型'] == '{self.rules['event_type']}',
    ((df_pd['委托价格'] == df_pd['bid1']) | (df_pd['委托价格'] == df_pd['ask1'])),
    df_pd['委托数量'] > df_pd.groupby('ticker')['委托数量'].transform('quantile', {self.rules['large_order_quantile']})
]
"""

class LayeringCancelPattern(SpoofingPattern):
    """分层撤单模式"""
    
    def __init__(self):
        super().__init__(
            name="layering_cancel",
            description="分层模式下的快速撤单",
            rules={
                "survival_time_ms": 1000,
                "event_type": "撤单",
                "requires_layering_score": True
            }
        )
    
    def generate_label_logic(self) -> str:
        return f"""
# 规则4: 分层撤单 - 分层模式下的快速撤单
conditions_r4 = [
    df_pd['layering_score'] > 0,
    df_pd['存活时间_ms'] < {self.rules['survival_time_ms']},
    df_pd['事件类型'] == '{self.rules['event_type']}'
]
"""

class ActiveHoursSpoofingPattern(SpoofingPattern):
    """活跃时段异常模式"""
    
    def __init__(self):
        super().__init__(
            name="active_hours_spoofing",
            description="开盘收盘时段的异常行为",
            rules={
                "survival_time_ms": 50,
                "event_type": "撤单",
                "active_hours": ["09:30-10:30", "14:00-15:00"],
                "size_multiplier": 1.0
            }
        )
    
    def generate_label_logic(self) -> str:
        return f"""
# 规则5: 活跃时段异常 - 开盘收盘时段的异常行为
market_active_hours = (
    (df_pd['委托_datetime'].dt.time >= pd.to_datetime('09:30').time()) &
    (df_pd['委托_datetime'].dt.time <= pd.to_datetime('10:30').time())
) | (
    (df_pd['委托_datetime'].dt.time >= pd.to_datetime('14:00').time()) &
    (df_pd['委托_datetime'].dt.time <= pd.to_datetime('15:00').time())
)
conditions_r5 = [
    df_pd['存活时间_ms'] < {self.rules['survival_time_ms']},
    df_pd['事件类型'] == '{self.rules['event_type']}',
    market_active_hours,
    df_pd['委托数量'] > df_pd.groupby('ticker')['委托数量'].transform('median') * {self.rules['size_multiplier']}
]
"""

class CompleteSpoofingPipeline:
    """完整的虚假报单检测Pipeline"""
    
    def __init__(self):
        self.patterns = [
            QuickCancelImpactPattern(),
            PriceManipulationPattern(),
            FakeLiquidityPattern(),
            LayeringCancelPattern(),
            ActiveHoursSpoofingPattern()
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
    
    def generate_extended_etl_code(self) -> str:
        """生成包含扩展标签的ETL代码"""
        
        # 收集所有模式的标签逻辑
        pattern_logics = []
        pattern_names = []
        
        for pattern in self.patterns:
            pattern_logics.append(pattern.generate_label_logic())
            pattern_names.append(pattern.name)
        
        etl_code = f'''
def improved_spoofing_rules(df_pd: pd.DataFrame) -> pd.DataFrame:
    """改进的欺诈检测规则 - Extended Labels"""
    labels = {{}}
    try:
        # 确保必要的列存在
        if 'bid1' not in df_pd.columns and '申买价1' in df_pd.columns: 
            df_pd['bid1'] = df_pd['申买价1']
        if 'ask1' not in df_pd.columns and '申卖价1' in df_pd.columns: 
            df_pd['ask1'] = df_pd['申卖价1']

        if 'at_bid' not in df_pd.columns and all(c in df_pd.columns for c in ['委托价格', 'bid1', '方向_委托']):
            df_pd['at_bid'] = ((df_pd['委托价格'] == df_pd['bid1']) & (df_pd['方向_委托'] == '买')).astype(int)
        elif 'at_bid' not in df_pd.columns:
            df_pd['at_bid'] = 0

        if 'at_ask' not in df_pd.columns and all(c in df_pd.columns for c in ['委托价格', 'ask1', '方向_委托']):
            df_pd['at_ask'] = ((df_pd['委托价格'] == df_pd['ask1']) & (df_pd['方向_委托'] == '卖')).astype(int)
        elif 'at_ask' not in df_pd.columns:
            df_pd['at_ask'] = 0
            
        df_pd['price_aggressiveness'] = df_pd.get('price_aggressiveness', 0.0)
        df_pd['layering_score'] = df_pd.get('layering_score', 0)
        df_pd['委托_datetime'] = pd.to_datetime(df_pd['委托_datetime'])
        
        {"".join(pattern_logics)}
        
        # 应用各个规则
        {"".join([f"labels['{pattern.name}'] = np.all(conditions_r{i+1}, axis=0).astype(int) if all(col in df_pd.columns for col in ['存活时间_ms', '事件类型', '委托数量', 'ticker']) else np.zeros(len(df_pd), dtype=int)" for i, pattern in enumerate(self.patterns)])}
        
        # 组合标签
        labels_df = pd.DataFrame(labels)
        
        # Extended Labels 组合策略
        df_pd['extended_liberal'] = (
            labels_df[{pattern_names}].sum(axis=1) >= 1
        ).astype(int)  # 任意一种模式
        
        df_pd['extended_moderate'] = (
            labels_df[{pattern_names}].sum(axis=1) >= 2  
        ).astype(int)  # 至少两种模式
        
        df_pd['extended_strict'] = (
            labels_df[{pattern_names}].sum(axis=1) >= 3
        ).astype(int)  # 至少三种模式
        
        # 原有的组合标签保持兼容
        df_pd['composite_spoofing'] = df_pd['extended_liberal']  # 等同于liberal
        df_pd['conservative_spoofing'] = df_pd['extended_moderate']  # 等同于moderate
        
        # 将各个模式标签也加入dataframe
        for col_name in labels_df.columns:
            if col_name not in df_pd.columns:
                df_pd[col_name] = labels_df[col_name]
                
    except Exception as e:
        print(f"  ⚠️ Error in improved_spoofing_rules: {{e}}")
        default_labels = {pattern_names} + ['extended_liberal', 'extended_moderate', 'extended_strict', 
                          'composite_spoofing', 'conservative_spoofing']
        for label_name in default_labels:
            if label_name not in df_pd.columns:
                df_pd[label_name] = 0
    return df_pd
'''
        return etl_code
    
    def run_merge_step(self, base_data_root: str, tickers: List[str] = None) -> bool:
        """步骤1: 运行数据合并"""
        print("🔗 Step 1: Merging order and trade data...")
        
        cmd = [
            sys.executable, "scripts/data_process/merge_order_trade.py",
            "--root", base_data_root
        ]
        
        if tickers:
            cmd.extend(["--tickers"] + tickers)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Data merge completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Data merge failed: {e.stderr}")
            return False
    
    def run_etl_step(self, event_stream_root: str, tickers: List[str] = None) -> bool:
        """步骤2: 运行ETL处理和扩展标签生成"""
        print("🔧 Step 2: Running ETL with Extended Labels...")
        
        cmd = [
            sys.executable, "scripts/data_process/run_etl_from_event.py",
            "--root", event_stream_root,
            "--enhanced_labels",  # 使用增强标签
            "--backend", "polars",
            "--max_workers", "100"
        ]
        
        if tickers:
            cmd.extend(["--tickers"] + tickers)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ ETL processing completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ ETL processing failed: {e.stderr}")
            return False
    
    def load_data(self, data_root: Path) -> pd.DataFrame:
        """加载和合并处理后的数据"""
        print("📥 Step 3: Loading processed data...")
        
        # 加载特征
        feat_files = list((data_root / "features_select").glob("X_*.parquet"))
        if not feat_files:
            raise FileNotFoundError("No feature files found. Please run ETL first.")
        
        df_features = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
        
        # 加载标签
        label_files = list((data_root / "labels_select").glob("labels_*.parquet"))
        if not label_files:
            raise FileNotFoundError("No label files found. Please run ETL first.")
        
        df_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
        
        # 合并数据
        df = df_features.merge(df_labels, on=['自然日', 'ticker', '交易所委托号'], how='inner')
        
        print(f"  Features: {df_features.shape}")
        print(f"  Labels: {df_labels.shape}")
        print(f"  Merged: {df.shape}")
        
        return df
    
    def analyze_extended_labels(self, df: pd.DataFrame) -> Dict:
        """分析扩展标签分布"""
        print("\n🏷️ Extended Labels Analysis:")
        
        results = {}
        
        # 原始标签
        if 'y_label' in df.columns:
            pos_count = int(df['y_label'].sum())  # 转换为Python int
            pos_rate = float(pos_count / len(df) * 100)  # 转换为Python float
            results['original'] = {'count': pos_count, 'rate': pos_rate}
            print(f"  {'Original y_label':<25}: {pos_count:>8,} ({pos_rate:>6.3f}%)")
        
        # 各种模式标签
        pattern_counts = []
        for pattern in self.patterns:
            if pattern.name in df.columns:
                pos_count = int(df[pattern.name].sum())  # 转换为Python int
                pos_rate = float(pos_count / len(df) * 100)  # 转换为Python float
                results[pattern.name] = {
                    'count': pos_count, 
                    'rate': pos_rate,
                    'description': pattern.description
                }
                print(f"  {pattern.name:<25}: {pos_count:>8,} ({pos_rate:>6.3f}%) - {pattern.description}")
                pattern_counts.append(pattern.name)
        
        # 🎯 动态生成Extended Labels组合标签
        if pattern_counts:  # 确保有可用的模式标签
            print(f"\n📊 Generating Extended Labels (Dynamic):")
            
            # Extended Liberal: 任意一种模式即为虚假报单
            pattern_cols = [col for col in pattern_counts if col in df.columns]
            if pattern_cols:
                df['extended_liberal'] = (df[pattern_cols].sum(axis=1) >= 1).astype(int)
                df['extended_moderate'] = (df[pattern_cols].sum(axis=1) >= 2).astype(int)
                df['extended_strict'] = (df[pattern_cols].sum(axis=1) >= 3).astype(int)
                
                # 扩展组合标签统计
                for ext_label in ['extended_liberal', 'extended_moderate', 'extended_strict']:
                    pos_count = int(df[ext_label].sum())  # 转换为Python int
                    pos_rate = float(pos_count / len(df) * 100)  # 转换为Python float
                    results[ext_label] = {'count': pos_count, 'rate': pos_rate}
                    
                    if ext_label == 'extended_liberal':
                        print(f"  🎯 {ext_label:<22}: {pos_count:>8,} ({pos_rate:>6.3f}%) [任意模式]")
                    else:
                        print(f"     {ext_label:<22}: {pos_count:>8,} ({pos_rate:>6.3f}%)")
        
        return results
    
    def train_evaluate_model(self, X_train, y_train, X_valid, y_valid, label_name: str) -> Dict:
        """训练和评估单个模型"""
        pos_count = y_train.sum()
        if pos_count == 0:
            print(f"  ⚠️ {label_name}: No positive samples")
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
        
        print(f"  {label_name:<25}: PR-AUC={metrics['PR-AUC']:.6f}, Precision@0.1%={metrics['Precision@0.1%']:.6f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred_proba
        }
    
    def run_training_evaluation(self, df: pd.DataFrame, train_regex: str, valid_regex: str, 
                               results_dir: Path, by_ticker: bool = False) -> Dict:
        """步骤4: 运行训练和评估"""
        print(f"\n🎯 Step 4: Training and Evaluation ({'by ticker' if by_ticker else 'combined'}):")
        
        # 准备特征
        leakage_features = [
            "final_survival_time_ms", "total_events", "total_traded_qty",
            "num_trades", "num_cancels", "is_fully_filled", "layering_score"
        ]
        
        df_clean = df.drop(columns=leakage_features, errors='ignore')
        available_features = [f for f in self.safe_features if f in df_clean.columns]
        
        print(f"  Using {len(available_features)} safe features")
        
        # 数据切分
        train_mask = df_clean["自然日"].astype(str).str.contains(train_regex)
        valid_mask = df_clean["自然日"].astype(str).str.contains(valid_regex)
        
        df_train = df_clean[train_mask].copy()
        df_valid = df_clean[valid_mask].copy()
        
        print(f"  Training: {len(df_train):,} samples")
        print(f"  Validation: {len(df_valid):,} samples")
        
        if by_ticker:
            return self._train_by_ticker(df_train, df_valid, available_features, results_dir)
        else:
            return self._train_combined(df_train, df_valid, available_features)
    
    def _train_combined(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, available_features: List[str]) -> Dict:
        """传统的合并训练方式"""
        X_train = df_train[available_features].fillna(0)
        X_valid = df_valid[available_features].fillna(0)
        
        # 训练所有标签策略
        all_results = {}
        
        print("  📊 Combined Training Results:")
        # 原始标签
        if 'y_label' in df_train.columns:
            result = self.train_evaluate_model(
                X_train, df_train['y_label'], X_valid, df_valid['y_label'], 'Original'
            )
            if result:
                all_results['original'] = result['metrics']
        
        # 各种虚假报单模式
        for pattern in self.patterns:
            if pattern.name in df_train.columns:
                result = self.train_evaluate_model(
                    X_train, df_train[pattern.name], X_valid, df_valid[pattern.name], pattern.name
                )
                if result:
                    all_results[pattern.name] = result['metrics']
        
        # 扩展标签组合
        for ext_label in ['extended_liberal', 'extended_moderate', 'extended_strict']:
            if ext_label in df_train.columns:
                result = self.train_evaluate_model(
                    X_train, df_train[ext_label], X_valid, df_valid[ext_label], ext_label
                )
                if result:
                    all_results[ext_label] = result['metrics']
        
        return all_results
    
    def _train_by_ticker(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, 
                        available_features: List[str], results_dir: Path) -> Dict:
        """按股票分开训练评估"""
        all_results = {}
        ticker_results = {}
        
        unique_tickers = sorted(df_train['ticker'].unique())
        print(f"  📈 Training separately for {len(unique_tickers)} tickers: {unique_tickers}")
        
        # 各个股票分别训练
        for ticker in unique_tickers:
            print(f"\n  🔸 Training for {ticker}:")
            
            # 分离当前股票数据
            train_ticker = df_train[df_train['ticker'] == ticker].copy()
            valid_ticker = df_valid[df_valid['ticker'] == ticker].copy()
            
            if len(train_ticker) == 0 or len(valid_ticker) == 0:
                print(f"    ⚠️ No data for {ticker}")
                continue
            
            print(f"    Train: {len(train_ticker):,}, Valid: {len(valid_ticker):,}")
            
            X_train_ticker = train_ticker[available_features].fillna(0)
            X_valid_ticker = valid_ticker[available_features].fillna(0)
            
            ticker_results[ticker] = {}
            
            # 训练各种标签
            labels_to_train = ['y_label'] + [p.name for p in self.patterns] + \
                            ['extended_liberal', 'extended_moderate', 'extended_strict']
            
            for label_name in labels_to_train:
                if label_name in train_ticker.columns:
                    result = self.train_evaluate_model(
                        X_train_ticker, train_ticker[label_name], 
                        X_valid_ticker, valid_ticker[label_name], 
                        f'{ticker}_{label_name}'
                    )
                    if result:
                        ticker_results[ticker][label_name] = result['metrics']
        
        # 计算平均性能和最佳股票
        all_results['by_ticker'] = ticker_results
        all_results['ticker_averages'] = self._compute_ticker_averages(ticker_results)
        all_results['best_performers'] = self._find_best_performers(ticker_results)
        
        # 保存分股票结果
        self._save_ticker_results(ticker_results, results_dir)
        
        return all_results
    
    def _compute_ticker_averages(self, ticker_results: Dict) -> Dict:
        """计算各标签在所有股票上的平均性能"""
        averages = {}
        
        # 收集所有标签
        all_labels = set()
        for ticker_data in ticker_results.values():
            all_labels.update(ticker_data.keys())
        
        for label in all_labels:
            label_metrics = {}
            for metric in ['PR-AUC', 'ROC-AUC', 'Precision@0.1%', 'Precision@1.0%']:
                values = []
                for ticker_data in ticker_results.values():
                    if label in ticker_data and metric in ticker_data[label]:
                        values.append(ticker_data[label][metric])
                
                if values:
                    label_metrics[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            if label_metrics:
                averages[label] = label_metrics
        
        return averages
    
    def _find_best_performers(self, ticker_results: Dict) -> Dict:
        """找到各标签的最佳表现股票"""
        best_performers = {}
        
        # 收集所有标签
        all_labels = set()
        for ticker_data in ticker_results.values():
            all_labels.update(ticker_data.keys())
        
        for label in all_labels:
            best_performers[label] = {}
            
            for metric in ['PR-AUC', 'ROC-AUC', 'Precision@0.1%']:
                best_score = -1
                best_ticker = None
                
                for ticker, ticker_data in ticker_results.items():
                    if label in ticker_data and metric in ticker_data[label]:
                        score = ticker_data[label][metric]
                        if score > best_score:
                            best_score = score
                            best_ticker = ticker
                
                if best_ticker:
                    best_performers[label][metric] = {
                        'ticker': best_ticker,
                        'score': best_score
                    }
        
        return best_performers
    
    def _save_ticker_results(self, ticker_results: Dict, results_dir: Path):
        """保存分股票结果"""
        # 创建详细的分股票结果表格
        rows = []
        for ticker, ticker_data in ticker_results.items():
            for label, metrics in ticker_data.items():
                row = {'ticker': ticker, 'label': label}
                row.update(metrics)
                rows.append(row)
        
        if rows:
            ticker_df = pd.DataFrame(rows)
            ticker_df.to_csv(results_dir / "by_ticker_results.csv", index=False, float_format='%.8f')
            
            # 创建透视表 - 更易读
            for metric in ['PR-AUC', 'ROC-AUC', 'Precision@0.1%']:
                if metric in ticker_df.columns:
                    pivot = ticker_df.pivot(index='ticker', columns='label', values=metric)
                    pivot.to_csv(results_dir / f"by_ticker_{metric.replace('@', '_at_').replace('%', 'pct')}.csv", 
                               float_format='%.6f')
    
    def generate_final_report(self, all_results: Dict, label_analysis: Dict, results_dir: Path, by_ticker: bool = False):
        """生成最终报告"""
        print("\n📊 Final Extended Labels Report")
        print("=" * 80)
        
        if by_ticker and 'by_ticker' in all_results:
            self._generate_ticker_report(all_results, results_dir)
        else:
            self._generate_combined_report(all_results, results_dir)
        
        # 标签分析
        with open(results_dir / "extended_labels_analysis.json", 'w') as f:
            json.dump(label_analysis, f, indent=2)
        
        with open(results_dir / "training_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 All results saved to: {results_dir}")
        
        # 🎯 Extended Liberal Strategy 总结
        if 'extended_liberal' in label_analysis:
            liberal_stats = label_analysis['extended_liberal']
            print(f"\n💡 Extended Liberal Strategy Summary:")
            print(f"  ✅ 识别了 {liberal_stats['count']:,} 个虚假报单 ({liberal_stats['rate']:.3f}%)")
            print(f"  ✅ 涵盖了 {len(self.patterns)} 种虚假报单模式")
            print(f"  ✅ 任意一种模式即为虚假报单 (最宽松策略)")
            print(f"  ✅ 为监管提供最全面的虚假报单检测")
        
        # 架构总结
        training_mode = "分股票训练" if by_ticker else "合并训练"
        print(f"\n🏗️ Extended Labels Architecture Summary ({training_mode}):")
        print(f"  ✅ 定义了 {len(self.patterns)} 种虚假报单模式")
        print(f"  ✅ 生成了多层次的扩展标签 (Liberal/Moderate/Strict)")
        print(f"  ✅ 严格控制特征无数据泄露")
        print(f"  ✅ 提供完整的端到端pipeline")
        print(f"  ✅ Extended Liberal = 主要展示结果")
        if by_ticker:
            print(f"  ✅ 分股票训练 = 更精准的个股模式识别")
    
    def _generate_combined_report(self, all_results: Dict, results_dir: Path):
        """生成合并训练的报告"""
        # 性能对比表格
        if all_results:
            comparison_df = pd.DataFrame(all_results).T
            
            key_metrics = ['PR-AUC', 'ROC-AUC', 'Precision@0.1%', 'Precision@1.0%']
            available_metrics = [m for m in key_metrics if m in comparison_df.columns]
            
            print(f"\n🎯 Performance Comparison (Combined Training):")
            print(comparison_df[available_metrics].round(6))
            
            # 🎯 重点展示Extended Liberal结果
            if 'extended_liberal' in comparison_df.index:
                liberal_results = comparison_df.loc['extended_liberal']
                print(f"\n🏆 Extended Liberal Strategy (任意一种模式即为虚假报单):")
                print(f"  📊 PR-AUC: {liberal_results['PR-AUC']:.6f}")
                print(f"  📊 ROC-AUC: {liberal_results['ROC-AUC']:.6f}")
                print(f"  📊 Precision@0.1%: {liberal_results['Precision@0.1%']:.6f}")
                print(f"  📊 Precision@1.0%: {liberal_results['Precision@1.0%']:.6f}")
                
                # 与原始标签对比
                if 'original' in comparison_df.index:
                    orig_pr_auc = comparison_df.loc['original', 'PR-AUC']
                    liberal_pr_auc = liberal_results['PR-AUC']
                    improvement = (liberal_pr_auc - orig_pr_auc) / orig_pr_auc * 100
                    print(f"  📈 vs Original: {improvement:+.1f}% improvement in PR-AUC")
            
            # 最佳策略
            if 'PR-AUC' in comparison_df.columns:
                best_strategy = comparison_df['PR-AUC'].idxmax()
                best_score = comparison_df.loc[best_strategy, 'PR-AUC']
                print(f"\n🥇 Overall Best Strategy: {best_strategy} (PR-AUC: {best_score:.6f})")
            
            # 保存结果
            comparison_df.to_csv(results_dir / "extended_labels_performance.csv", float_format='%.8f')
    
    def _generate_ticker_report(self, all_results: Dict, results_dir: Path):
        """生成分股票训练的报告"""
        ticker_averages = all_results.get('ticker_averages', {})
        best_performers = all_results.get('best_performers', {})
        
        print(f"\n🎯 Performance Summary (By Ticker Training):")
        
        # 平均性能表格
        if ticker_averages:
            avg_data = []
            for label, metrics in ticker_averages.items():
                row = {'Label': label}
                for metric, stats in metrics.items():
                    row[f'{metric}_mean'] = stats['mean']
                    row[f'{metric}_std'] = stats['std']
                avg_data.append(row)
            
            if avg_data:
                avg_df = pd.DataFrame(avg_data)
                key_cols = ['Label'] + [col for col in avg_df.columns if 'PR-AUC' in col or 'Precision@0.1%' in col]
                available_cols = [col for col in key_cols if col in avg_df.columns]
                
                print(avg_df[available_cols].round(6))
                avg_df.to_csv(results_dir / "ticker_averages.csv", index=False, float_format='%.8f')
        
        # 🎯 重点展示Extended Liberal平均结果
        if 'extended_liberal' in ticker_averages:
            liberal_avg = ticker_averages['extended_liberal']
            if 'PR-AUC' in liberal_avg:
                print(f"\n🏆 Extended Liberal Strategy Average Performance:")
                print(f"  📊 PR-AUC: {liberal_avg['PR-AUC']['mean']:.6f} ± {liberal_avg['PR-AUC']['std']:.6f}")
                print(f"  📊 Range: [{liberal_avg['PR-AUC']['min']:.6f}, {liberal_avg['PR-AUC']['max']:.6f}]")
                
                if 'Precision@0.1%' in liberal_avg:
                    print(f"  📊 Precision@0.1%: {liberal_avg['Precision@0.1%']['mean']:.6f} ± {liberal_avg['Precision@0.1%']['std']:.6f}")
        
        # 最佳表现股票
        if best_performers:
            print(f"\n🥇 Best Performing Tickers:")
            for label in ['extended_liberal', 'y_label']:
                if label in best_performers:
                    label_best = best_performers[label]
                    if 'PR-AUC' in label_best:
                        ticker = label_best['PR-AUC']['ticker']
                        score = label_best['PR-AUC']['score']
                        print(f"  {label:<25}: {ticker} (PR-AUC: {score:.6f})")
        
        # 性能分布分析
        if 'by_ticker' in all_results:
            self._analyze_ticker_performance_distribution(all_results['by_ticker'], results_dir)
    
    def _analyze_ticker_performance_distribution(self, ticker_results: Dict, results_dir: Path):
        """分析分股票性能分布"""
        print(f"\n📈 Ticker Performance Distribution Analysis:")
        
        # 关键标签分析
        key_labels = ['extended_liberal', 'y_label']
        
        for label in key_labels:
            if any(label in ticker_data for ticker_data in ticker_results.values()):
                pr_aucs = []
                tickers = []
                
                for ticker, ticker_data in ticker_results.items():
                    if label in ticker_data and 'PR-AUC' in ticker_data[label]:
                        pr_aucs.append(ticker_data[label]['PR-AUC'])
                        tickers.append(ticker)
                
                if pr_aucs:
                    pr_aucs = np.array(pr_aucs)
                    print(f"\n  {label} PR-AUC Distribution:")
                    print(f"    📊 平均值: {pr_aucs.mean():.6f}")
                    print(f"    📊 标准差: {pr_aucs.std():.6f}")
                    print(f"    📊 最小值: {pr_aucs.min():.6f} ({tickers[np.argmin(pr_aucs)]})")
                    print(f"    📊 最大值: {pr_aucs.max():.6f} ({tickers[np.argmax(pr_aucs)]})")
                    print(f"    📊 中位数: {np.median(pr_aucs):.6f}")
                    
                    # 性能分档
                    high_performers = [(tickers[i], pr_aucs[i]) for i in range(len(pr_aucs)) if pr_aucs[i] > pr_aucs.mean() + pr_aucs.std()]
                    low_performers = [(tickers[i], pr_aucs[i]) for i in range(len(pr_aucs)) if pr_aucs[i] < pr_aucs.mean() - pr_aucs.std()]
                    
                    if high_performers:
                        print(f"    🏆 高性能股票: {high_performers}")
                    if low_performers:
                        print(f"    ⚠️ 低性能股票: {low_performers}")
    
    def run_complete_pipeline(self, base_data_root: str, train_regex: str, valid_regex: str, 
                            tickers: List[str] = None, results_dir: str = None, 
                            skip_merge: bool = False, skip_etl: bool = False, by_ticker: bool = False):
        """运行完整pipeline"""
        print("🚀 Complete Spoofing Detection Pipeline with Extended Labels")
        print("=" * 80)
        
        base_path = Path(base_data_root)
        event_stream_root = base_path.parent / "event_stream"
        data_root = base_path.parent
        
        if results_dir:
            results_dir = Path(results_dir)
        else:
            results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)
        
        print(f"📁 Base data: {base_path}")
        print(f"📁 Event stream: {event_stream_root}")
        print(f"📁 Results: {results_dir}")
        
        success = True
        
        # 步骤1: 数据合并
        if not skip_merge:
            success = self.run_merge_step(str(base_path), tickers)
            if not success:
                return None
        
        # 步骤2: ETL处理
        if not skip_etl:
            success = self.run_etl_step(str(event_stream_root), tickers)
            if not success:
                return None
        
        # 步骤3: 加载数据
        try:
            df = self.load_data(data_root)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return None
        
        # 步骤4: 分析扩展标签
        label_analysis = self.analyze_extended_labels(df)
        
        # 步骤5: 训练评估
        all_results = self.run_training_evaluation(df, train_regex, valid_regex, results_dir, by_ticker)
        
        # 步骤6: 生成报告
        self.generate_final_report(all_results, label_analysis, results_dir, by_ticker)
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Complete Spoofing Detection Pipeline with Extended Labels")
    parser.add_argument("--base_data_root", required=True, help="原始数据根目录 (base_data)")
    parser.add_argument("--tickers", nargs="*", help="股票列表")
    parser.add_argument("--train_regex", default="202503|202504", help="训练数据日期正则")
    parser.add_argument("--valid_regex", default="202505", help="验证数据日期正则")
    parser.add_argument("--results_dir", help="结果保存目录")
    parser.add_argument("--skip_merge", action="store_true", help="跳过数据合并步骤")
    parser.add_argument("--skip_etl", action="store_true", help="跳过ETL处理步骤")
    parser.add_argument("--by_ticker", action="store_true", help="按股票分开训练评估")
    
    args = parser.parse_args()
    
    # 运行完整pipeline
    pipeline = CompleteSpoofingPipeline()
    results = pipeline.run_complete_pipeline(
        base_data_root=args.base_data_root,
        train_regex=args.train_regex,
        valid_regex=args.valid_regex,
        tickers=args.tickers,
        results_dir=args.results_dir,
        skip_merge=args.skip_merge,
        skip_etl=args.skip_etl,
        by_ticker=args.by_ticker
    )

if __name__ == "__main__":
    main()

"""
使用示例：

# 1. 完整运行（推荐）
python complete_spoofing_pipeline.py \
  --base_data_root "/obs/users/fenglang/general/Spoofing Detect/data/base_data" \
  --tickers 000989.SZ 300233.SZ --by_ticker

# 2. 跳过前置步骤，直接训练
python complete_spoofing_pipeline.py \
  --base_data_root "/obs/users/fenglang/general/Spoofing Detect/data/base_data" \
  --skip_merge --skip_etl --by_ticker

# 3. 合并训练（对比用）
python complete_spoofing_pipeline.py \
  --base_data_root "/obs/users/fenglang/general/Spoofing Detect/data/base_data" \
  --skip_merge --skip_etl
""" 