#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced LightGBM Training Pipeline for Spoofing Detection (Fixed)
------------------------------------------------------------------
主要优化：
• 增强特征工程：基于现有列的技术指标、统计特征、交互特征
• 改进数据平衡策略：下采样 + 分层采样
• 模型集成：多种算法组合
• 更细致的超参数调优
• 增强评估指标和可视化
• 适配新的数据架构（符合data.md规范）
"""

import argparse, glob, os, re, sys, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ---------- Focal Loss Implementation --------------------------------
def focal_loss_objective(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for LightGBM
    Args:
        y_true: 真实标签
        y_pred: 预测概率（logits）
        alpha: 平衡因子，用于平衡正负样本
        gamma: focusing参数，用于减少易分类样本的权重
    """
    # 将logits转换为概率
    p = 1 / (1 + np.exp(-y_pred))
    
    # 计算focal loss
    ce_loss = -y_true * np.log(p + 1e-8) - (1 - y_true) * np.log(1 - p + 1e-8)
    p_t = p * y_true + (1 - p) * (1 - y_true)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    
    focal_weight = alpha_t * (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss
    
    # 计算梯度和海塞矩阵
    grad = focal_weight * (p - y_true)
    hess = focal_weight * p * (1 - p) * (gamma * (y_true - p) + 1)
    
    return grad, hess

def focal_loss_lgb(alpha=0.25, gamma=2.0):
    """返回LightGBM可用的focal loss目标函数"""
    def objective(y_true, y_pred):
        return focal_loss_objective(y_true, y_pred, alpha, gamma)
    return objective

# ---------- Enhanced Feature Engineering --------------------------------
def create_technical_indicators(df):
    """创建技术指标特征（基于现有列）"""
    features = {}
    
    # 价格技术指标
    features['price_volatility'] = df.groupby('ticker')['mid_price'].transform(
        lambda x: x.rolling(100, min_periods=1).std()
    )
    features['price_momentum'] = df.groupby('ticker')['mid_price'].transform(
        lambda x: x.pct_change(5)
    )
    features['relative_spread'] = df['spread'] / df['mid_price']
    features['spread_volatility'] = df.groupby('ticker')['spread'].transform(
        lambda x: x.rolling(50, min_periods=1).std()
    )
    
    # 订单流指标
    features['order_imbalance'] = (df['bid1'] - df['ask1']) / (df['bid1'] + df['ask1'])
    
    # 价格级别特征（基于现有的价格信息）
    # 假设委托价格可以通过mid_price和delta_mid重构
    features['at_bid'] = (df['delta_mid'] <= -df['spread']/2).astype(int)
    features['at_ask'] = (df['delta_mid'] >= df['spread']/2).astype(int)
    features['between_quotes'] = ((df['delta_mid'] > -df['spread']/2) & 
                                 (df['delta_mid'] < df['spread']/2)).astype(int)
    
    return pd.DataFrame(features)

def create_statistical_features(df):
    """创建统计特征"""
    features = {}
    
    # 分位数特征
    for col in ['log_qty', 'spread', 'delta_mid']:
        if col in df.columns:
            # 计算分位数rank
            features[f'{col}_rank'] = df.groupby('ticker')[col].transform(
                lambda x: x.rank(pct=True)
            )
            # 相对于市场平均的标准化
            features[f'{col}_zscore'] = df.groupby('ticker')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    
    # 历史行为特征（简化版，避免时间列问题）
    features['hist_order_size_mean'] = df.groupby('ticker')['log_qty'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    features['hist_spread_mean'] = df.groupby('ticker')['spread'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    return pd.DataFrame(features)

def create_interaction_features(df):
    """创建交互特征"""
    features = {}
    
    # 确保relative_spread存在
    if 'relative_spread' not in df.columns:
        df['relative_spread'] = df['spread'] / df['mid_price']
    
    # 关键交互
    features['qty_spread_interaction'] = df['log_qty'] * df['relative_spread']
    features['time_qty_interaction'] = df['time_cos'] * df['log_qty']
    features['direction_spread_interaction'] = df['is_buy'] * df['pct_spread']
    
    # 价格-时间交互
    features['price_time_interaction'] = df['price_dev_prevclose'] * df['time_sin']
    
    # 订单活动交互
    features['orders_cancels_ratio'] = df['orders_100ms'] / (df['cancels_5s'] + 1)
    features['activity_spread_interaction'] = df['orders_100ms'] * df['relative_spread']
    
    return pd.DataFrame(features)

def enhance_features(df):
    """增强特征工程"""
    print("🔧 Creating enhanced features...")
    print(f"Original columns: {df.columns.tolist()}")
    
    # 计算基础衍生特征
    if 'relative_spread' not in df.columns:
        df['relative_spread'] = df['spread'] / df['mid_price']
    
    # 创建技术指标
    try:
        tech_features = create_technical_indicators(df)
        print(f"Created {len(tech_features.columns)} technical indicators")
    except Exception as e:
        print(f"Warning: Technical indicators failed: {e}")
        tech_features = pd.DataFrame()
    
    # 创建统计特征
    try:
        stat_features = create_statistical_features(df)
        print(f"Created {len(stat_features.columns)} statistical features")
    except Exception as e:
        print(f"Warning: Statistical features failed: {e}")
        stat_features = pd.DataFrame()
    
    # 创建交互特征
    try:
        interaction_features = create_interaction_features(df)
        print(f"Created {len(interaction_features.columns)} interaction features")
    except Exception as e:
        print(f"Warning: Interaction features failed: {e}")
        interaction_features = pd.DataFrame()
    
    # 合并所有特征
    feature_dfs = [df]
    if not tech_features.empty:
        # 移除已存在的列
        tech_features = tech_features.loc[:, ~tech_features.columns.isin(df.columns)]
        if not tech_features.empty:
            feature_dfs.append(tech_features)
    if not stat_features.empty:
        stat_features = stat_features.loc[:, ~stat_features.columns.isin(df.columns)]
        if not stat_features.empty:
            feature_dfs.append(stat_features)
    if not interaction_features.empty:
        # 检查交互特征是否重复
        existing_cols = df.columns.tolist()
        if len(feature_dfs) > 1:
            for feat_df in feature_dfs[1:]:
                existing_cols.extend(feat_df.columns.tolist())
        interaction_features = interaction_features.loc[:, ~interaction_features.columns.isin(existing_cols)]
        if not interaction_features.empty:
            feature_dfs.append(interaction_features)
    
    enhanced_df = pd.concat(feature_dfs, axis=1)
    
    # 最终检查：移除任何重复的列
    enhanced_df = enhanced_df.loc[:, ~enhanced_df.columns.duplicated()]
    
    # 处理无穷值和缺失值
    enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
    enhanced_df = enhanced_df.fillna(0)
    
    print(f"Enhanced dataframe shape: {enhanced_df.shape}")
    print(f"Final columns: {len(enhanced_df.columns)} (duplicates removed)")
    return enhanced_df

# ---------- Advanced Sampling Strategies --------------------------------
def advanced_sampling(X, y, method='undersample', sampling_strategy='auto'):
    """高级采样策略（移除SMOTE，数据太不平衡）"""
    print(f"🎯 Applying {method} sampling...")
    
    original_pos = y.sum()
    original_neg = len(y) - original_pos
    print(f"Original distribution: {original_pos:,} positive, {original_neg:,} negative")
    
    if method == 'undersample':
        # 下采样负样本，保持合理比例
        pos_indices = y[y == 1].index
        neg_indices = y[y == 0].index
        
        # 保持1:10的比例，避免过度不平衡
        target_neg_size = min(len(pos_indices) * 10, len(neg_indices))
        selected_neg_indices = np.random.choice(neg_indices, target_neg_size, replace=False)
        
        selected_indices = np.concatenate([pos_indices, selected_neg_indices])
        selected_X = X.loc[selected_indices]
        selected_y = y.loc[selected_indices]
        
        new_pos = selected_y.sum()
        new_neg = len(selected_y) - new_pos
        print(f"After {method}: {new_pos:,} positive, {new_neg:,} negative")
        return selected_X, selected_y
        
    elif method == 'stratified_undersample':
        # 分层下采样，按股票分别采样
        pos_indices = y[y == 1].index
        neg_indices = y[y == 0].index
        
        # 每个股票保持相同的正负样本比例
        if 'ticker' in X.columns:
            selected_indices = []
            for ticker in X['ticker'].unique():
                ticker_mask = X['ticker'] == ticker
                ticker_pos = pos_indices[X.loc[pos_indices, 'ticker'] == ticker]
                ticker_neg = neg_indices[X.loc[neg_indices, 'ticker'] == ticker]
                
                if len(ticker_pos) > 0 and len(ticker_neg) > 0:
                    # 每个股票保持1:10比例
                    target_neg = min(len(ticker_pos) * 10, len(ticker_neg))
                    selected_neg = np.random.choice(ticker_neg, target_neg, replace=False)
                    selected_indices.extend(ticker_pos.tolist())
                    selected_indices.extend(selected_neg.tolist())
            
            selected_X = X.loc[selected_indices]
            selected_y = y.loc[selected_indices]
        else:
            # 回退到普通下采样
            return advanced_sampling(X, y, method='undersample')
        
        new_pos = selected_y.sum()
        new_neg = len(selected_y) - new_pos
        print(f"After {method}: {new_pos:,} positive, {new_neg:,} negative")
        return selected_X, selected_y
    
    else:
        # 不采样，使用原始数据
        print("Using original data without sampling")
        return X, y

# ---------- Model Ensemble -----------------------------------------------
class EnsembleClassifier:
    def __init__(self, models=None):
        if models is None:
            self.models = {
                'lgb': lgb.LGBMClassifier(
                    objective='binary',
                    metric='average_precision',
                    boosting_type='gbdt',
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=10,
                    reg_lambda=10,
                    random_state=42,
                    verbose=-1
                ),
                'xgb': XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='aucpr',
                    n_estimators=500,  # 减少轮数避免过拟合
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=10,
                    reg_lambda=10,
                    random_state=42,
                    verbosity=0
                ),
                'rf': RandomForestClassifier(
                    n_estimators=300,  # 减少树的数量
                    max_depth=8,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                )
            }
        else:
            self.models = models
        
        self.weights = None
        self.fitted_models = {}
    
    def fit(self, X, y, X_val=None, y_val=None):
        """训练集成模型"""
        print("🚀 Training ensemble models...")
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            if name == 'lgb' and X_val is not None:
                # LightGBM with early stopping
                try:
                    model.fit(
                        X, y,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=100,
                        verbose=False
                    )
                except TypeError:
                    from lightgbm import early_stopping
                    model.fit(
                        X, y,
                        eval_set=[(X_val, y_val)],
                        callbacks=[early_stopping(100)]
                    )
            else:
                # XGBoost和RandomForest直接训练
                model.fit(X, y)
            
            self.fitted_models[name] = model
        
        # 如果有验证集，计算权重
        if X_val is not None and y_val is not None:
            self._compute_weights(X_val, y_val)
        else:
            # 等权重
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
    
    def _compute_weights(self, X_val, y_val):
        """基于验证集性能计算权重"""
        scores = {}
        for name, model in self.fitted_models.items():
            pred_proba = model.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, pred_proba)
            scores[name] = score
            print(f"  {name} PR-AUC: {score:.4f}")
        
        # 基于性能计算权重
        total_score = sum(scores.values())
        if total_score > 0:
            self.weights = {name: score/total_score for name, score in scores.items()}
        else:
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        print(f"  Weights: {self.weights}")
    
    def predict_proba(self, X):
        """集成预测"""
        predictions = []
        for name, model in self.fitted_models.items():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred * self.weights[name])
        
        ensemble_pred = np.sum(predictions, axis=0)
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

# ---------- Hyperparameter Optimization ----------------------------------
def optimize_lgb_params(X_train, y_train, X_val, y_val, n_trials=50):
    """优化LightGBM超参数"""
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params, n_estimators=1000)
        
        # 修复early_stopping参数兼容性
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )
        except TypeError:
            # 处理新版本LightGBM
            from lightgbm import early_stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(100)]
            )
        
        pred_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, pred_proba)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best PR-AUC: {study.best_value:.6f}")
    return study.best_params

# ---------- Ticker Selection Functions -----------------------------------
def filter_tickers_by_criteria(df, method="random", max_tickers=None):
    """根据标准筛选股票"""
    if method == "random":
        # 随机选择
        available_tickers = df['ticker'].unique()
        if max_tickers and len(available_tickers) > max_tickers:
            selected_tickers = np.random.choice(available_tickers, max_tickers, replace=False)
            return selected_tickers.tolist()
        return available_tickers.tolist()
    
    elif method == "by_volume":
        # 按交易量选择（如果有相关字段）
        if 'log_qty' in df.columns:
            ticker_volumes = df.groupby('ticker')['log_qty'].sum().sort_values(ascending=False)
        else:
            print("⚠️ 没有交易量字段，使用样本数量作为代理")
            ticker_volumes = df.groupby('ticker').size().sort_values(ascending=False)
        
        if max_tickers:
            return ticker_volumes.head(max_tickers).index.tolist()
        return ticker_volumes.index.tolist()
    
    elif method == "by_positive_samples":
        # 按正样本数量选择
        if 'y_label' in df.columns:
            ticker_positive = df[df['y_label'] == 1].groupby('ticker').size().sort_values(ascending=False)
            if max_tickers:
                return ticker_positive.head(max_tickers).index.tolist()
            return ticker_positive.index.tolist()
        else:
            print("⚠️ 没有标签字段，回退到随机选择")
            return filter_tickers_by_criteria(df, "random", max_tickers)
    
    else:
        return df['ticker'].unique().tolist()

def apply_ticker_filtering(df, args):
    """应用股票筛选逻辑"""
    print(f"🔍 原始股票数量: {df['ticker'].nunique()}")
    print(f"原始数据量: {len(df):,} 条")
    
    # 1. 从文件读取股票列表
    if args.ticker_file:
        if os.path.exists(args.ticker_file):
            with open(args.ticker_file, 'r') as f:
                file_tickers = [line.strip() for line in f if line.strip()]
            print(f"📋 从文件读取到 {len(file_tickers)} 个股票代码")
            df = df[df['ticker'].isin(file_tickers)]
        else:
            print(f"⚠️ 股票文件不存在: {args.ticker_file}")
    
    # 2. 应用包含列表
    if args.include_tickers:
        print(f"📌 筛选指定股票: {args.include_tickers}")
        df = df[df['ticker'].isin(args.include_tickers)]
        missing_tickers = set(args.include_tickers) - set(df['ticker'].unique())
        if missing_tickers:
            print(f"⚠️ 以下股票在数据中未找到: {missing_tickers}")
    
    # 3. 应用排除列表
    if args.exclude_tickers:
        print(f"🚫 排除股票: {args.exclude_tickers}")
        df = df[~df['ticker'].isin(args.exclude_tickers)]
    
    # 4. 应用数量限制和选择方法
    if args.max_tickers:
        available_tickers = df['ticker'].unique()
        if len(available_tickers) > args.max_tickers:
            print(f"🎯 使用 {args.ticker_selection_method} 方法选择 {args.max_tickers} 个股票")
            selected_tickers = filter_tickers_by_criteria(df, args.ticker_selection_method, args.max_tickers)
            df = df[df['ticker'].isin(selected_tickers)]
            print(f"✅ 选中股票: {selected_tickers}")
    
    print(f"🔍 筛选后股票数量: {df['ticker'].nunique()}")
    print(f"筛选后数据量: {len(df):,} 条")
    
    # 显示股票分布统计
    if df['ticker'].nunique() <= 20:
        ticker_stats = df.groupby('ticker').agg({
            'y_label': ['count', 'sum']  # 总样本数和正样本数
        }).round(3)
        ticker_stats.columns = ['样本数', '正样本数']
        ticker_stats['正样本率%'] = (ticker_stats['正样本数'] / ticker_stats['样本数'] * 100).round(3)
        print("\n📊 股票分布统计:")
        for ticker, stats in ticker_stats.iterrows():
            print(f"  {ticker}: {stats['样本数']:,} 样本, {stats['正样本数']:,} 正样本 ({stats['正样本率%']:.3f}%)")
    else:
        # 股票太多，只显示汇总统计
        ticker_stats = df.groupby('ticker')['y_label'].agg(['count', 'sum'])
        print(f"\n📊 股票分布汇总:")
        print(f"  平均样本数: {ticker_stats['count'].mean():.0f}")
        print(f"  平均正样本数: {ticker_stats['sum'].mean():.1f}")
        print(f"  平均正样本率: {(ticker_stats['sum']/ticker_stats['count']).mean()*100:.3f}%")
    
    return df

# ---------- Enhanced Evaluation ------------------------------------------
def comprehensive_evaluation(y_true, y_pred_proba, y_pred_binary=None):
    """综合评估"""
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # 基础指标
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # 计算样本分布
    total_samples = len(y_true)
    positive_samples = y_true.sum()
    
    # Precision at K
    metrics = {'PR-AUC': pr_auc, 'ROC-AUC': roc_auc}
    
    print(f"\n📊 样本分布:")
    print(f"总样本数: {total_samples:,}")
    print(f"正样本数: {positive_samples:,} ({positive_samples/total_samples*100:.3f}%)")
    print(f"\n📈 Precision@K% (实际值 / 理论最大值 = 达成率):")
    
    for k in [0.001, 0.005, 0.01, 0.05]:
        k_int = max(1, int(len(y_true) * k))
        top_k_idx = y_pred_proba.argsort()[::-1][:k_int]
        prec_k = y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()
        
        # 计算理论最大值
        theoretical_max = min(positive_samples / k_int, 1.0)
        achievement_rate = prec_k / theoretical_max if theoretical_max > 0 else 0
        
        metrics[f'Precision@{k*100:.1f}%'] = prec_k
        metrics[f'Precision@{k*100:.1f}%_max'] = theoretical_max
        metrics[f'Precision@{k*100:.1f}%_achievement'] = achievement_rate
        
        print(f"  K={k*100:4.1f}%: {prec_k:.6f} / {theoretical_max:.6f} = {achievement_rate*100:5.1f}%")
    
    return metrics

# ---------- Main Function ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_regex", default="202503|202504")
    parser.add_argument("--valid_regex", default="202505")
    parser.add_argument("--sampling_method", choices=["none", "undersample", "stratified_undersample"], 
                       default="undersample", help="采样方法（移除SMOTE）")
    parser.add_argument("--use_ensemble", action="store_true", help="使用模型集成")
    parser.add_argument("--optimize_params", action="store_true", help="优化超参数")
    parser.add_argument("--n_trials", type=int, default=50, help="超参数优化试验次数")
    parser.add_argument("--use_enhanced_labels", action="store_true", 
                       help="使用增强标签而不是原始y_label")
    parser.add_argument("--label_type", choices=["composite_spoofing", "conservative_spoofing"], 
                       default="composite_spoofing",
                       help="使用哪种增强标签类型")
    parser.add_argument("--use_focal_loss", action="store_true", help="使用Focal Loss处理不平衡数据")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal Loss alpha参数")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma参数")
    parser.add_argument("--use_class_weight", action="store_true", help="使用类别权重处理不平衡数据")
    parser.add_argument("--class_weight_ratio", type=float, default=None, help="正样本权重比例，默认为负样本数/正样本数")
    parser.add_argument("--eval_output_dir", type=str, default=None, 
                       help="评估结果保存目录，默认为 results/evaluation_results")
    
    # 股票筛选相关参数
    parser.add_argument("--include_tickers", type=str, nargs="*", default=None,
                       help="指定包含的股票代码列表，例如: --include_tickers 000001.SZ 000002.SZ")
    parser.add_argument("--exclude_tickers", type=str, nargs="*", default=None,
                       help="指定排除的股票代码列表，例如: --exclude_tickers 000001.SZ 000002.SZ")
    parser.add_argument("--ticker_file", type=str, default=None,
                       help="从文件读取股票代码列表，每行一个股票代码")
    parser.add_argument("--max_tickers", type=int, default=None,
                       help="最大股票数量限制，随机选择指定数量的股票")
    parser.add_argument("--ticker_selection_method", choices=["random", "by_volume", "by_positive_samples"], 
                       default="random", help="股票选择方法：随机、按交易量、按正样本数")
    
    args = parser.parse_args()
    
    t0 = time.time()
    print("🔍 Loading and preparing data...")
    
    # Load data - 适配新的数据架构（符合data.md）
    # 新架构使用 features/ 和 labels/ 而不是 features_select/ 和 labels_select/
    feat_pats = [os.path.join(args.data_root, "features", "X_*.parquet")]
    lab_pats = [os.path.join(args.data_root, "labels", "labels_*.parquet")]
    
    # 如果新路径不存在，回退到旧路径以保持兼容性
    if not glob.glob(feat_pats[0]):
        print("⚠️ New architecture paths not found, trying legacy paths...")
        feat_pats = [os.path.join(args.data_root, "features_select", "X_*.parquet")]
        lab_pats = [os.path.join(args.data_root, "labels_select", "labels_*.parquet")]
    
    # 加载特征数据
    files = []
    for pat in feat_pats:
        files.extend(sorted(glob.glob(pat)))
    
    if not files:
        print(f"❌ No feature files found. Tried:")
        print(f"  New: {args.data_root}/features/X_*.parquet")
        print(f"  Legacy: {args.data_root}/features_select/X_*.parquet")
        print("Please run the data processing pipeline first:")
        print("python scripts/data_process/complete_pipeline.py --base_data_dir <path> --output_root <path>")
        return
    
    print(f"📊 Loading features from {len(files)} files...")
    df_feat = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Features data shape: {df_feat.shape}")
    print(f"Feature columns: {df_feat.columns.tolist()}")
    
    # 加载标签数据
    files = []
    for pat in lab_pats:
        files.extend(sorted(glob.glob(pat)))
    
    # 如果标准位置没有找到标签，尝试labels_enhanced目录
    if not files:
        enhanced_lab_pat = os.path.join(args.data_root, "labels_enhanced", "labels_*.parquet")
        files.extend(sorted(glob.glob(enhanced_lab_pat)))
        if files:
            print(f"✅ Found {len(files)} label files in labels_enhanced directory")
    
    if not files:
        print(f"❌ No label files found. Tried:")
        print(f"  New: {args.data_root}/labels/labels_*.parquet")
        print(f"  Legacy: {args.data_root}/labels_select/labels_*.parquet")
        print(f"  Enhanced: {args.data_root}/labels_enhanced/labels_*.parquet")
        print("Please run the data processing pipeline first.")
        return
    
    print(f"📊 Loading labels from {len(files)} files...")
    df_lab = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Labels data shape: {df_lab.shape}")
    print(f"Label columns: {df_lab.columns.tolist()}")
    
    # 检查增强标签是否可用
    if args.use_enhanced_labels:
        if args.label_type not in df_lab.columns:
            print(f"❌ Enhanced label '{args.label_type}' not found in labels.")
            print(f"Available label columns: {df_lab.columns.tolist()}")
            print("Please ensure ETL was run with --enhanced_labels flag.")
            return
        print(f"✅ Using enhanced label: {args.label_type}")
    
    # 合并数据
    df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="inner")
    print(f"Merged data shape: {df.shape}")
    print(f"Merged columns: {df.columns.tolist()}")
    
    # 设置目标变量
    if args.use_enhanced_labels:
        df['y_label'] = df[args.label_type]
        print(f"Using {args.label_type} as target variable")
        print(f"Target distribution: {df['y_label'].value_counts().to_dict()}")
    else:
        if 'y_label' not in df.columns:
            print("❌ y_label column not found in data")
            return
        print("Using original y_label as target variable")
        print(f"Target distribution: {df['y_label'].value_counts().to_dict()}")
    
    # 应用股票筛选
    df = apply_ticker_filtering(df, args)
    
    # Split data
    train_mask = df["自然日"].astype(str).str.contains(args.train_regex)
    valid_mask = df["自然日"].astype(str).str.contains(args.valid_regex)
    
    df_train = df[train_mask].copy()
    df_valid = df[valid_mask].copy()
    
    print(f"Training: {len(df_train):,}, Validation: {len(df_valid):,}")
    
    # Enhanced feature engineering
    df_train = enhance_features(df_train)
    df_valid = enhance_features(df_valid)
    
    # 使用明确定义的列，避免数据泄露
    print("🛡️ 使用明确定义的特征列...")
    
    # 导入列定义模块
    try:
        import sys
        from pathlib import Path
        
        # 导入特征列定义
        sys.path.append(str(Path(__file__).parent.parent / "data_process" / "features"))
        from feature_generator import get_training_feature_columns, get_key_columns, get_leakage_risk_columns
        
        # 导入标签列定义
        sys.path.append(str(Path(__file__).parent.parent / "data_process" / "labels"))
        from label_generator import get_training_target_columns, get_label_columns
        
        # 获取可用于训练的特征列
        feature_cols = get_training_feature_columns(include_enhanced=True)
        
        # 获取实际存在的特征列
        available_cols = df_train.columns.tolist()
        actual_feature_cols = [col for col in feature_cols if col in available_cols]
        
        # 检查缺失的重要特征
        missing_features = [col for col in feature_cols if col not in available_cols]
        if missing_features:
            print(f"⚠️ 缺失 {len(missing_features)} 个定义的特征: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
        
        # 检查可能的泄露风险列
        leakage_risk_cols = get_leakage_risk_columns()
        potential_leakage = [col for col in available_cols if col in leakage_risk_cols]
        if potential_leakage:
            print(f"🚨 检测到潜在泄露风险列: {potential_leakage}")
            # 从特征列中移除
            actual_feature_cols = [col for col in actual_feature_cols if col not in potential_leakage]
        
        feature_cols = actual_feature_cols
        print(f"✅ 使用 {len(feature_cols)} 个安全特征")
        
        # 显示特征类别统计
        if len(feature_cols) > 0:
            base_features = [col for col in feature_cols if col in available_cols and 
                           any(base_col in col for base_col in ['mid_price', 'spread', 'log_qty', 'is_buy', 'time_', 'orders_', 'book_'])]
            enhanced_features = [col for col in feature_cols if col not in base_features]
            print(f"  基础特征: {len(base_features)}, 扩展特征: {len(enhanced_features)}")
        
    except ImportError as e:
        print(f"⚠️ 无法导入列定义模块，使用基础清理: {e}")
        
        # 基础清理：手动排除已知泄露特征
        leakage_cols = [
            "存活时间_ms", "事件_datetime", "成交价格", "成交数量", "事件类型",
            "is_cancel_event", "is_trade_event",
            "flag_R1", "flag_R2", "enhanced_spoofing_liberal", 
            "enhanced_spoofing_moderate", "enhanced_spoofing_strict"
        ]
        
        id_cols = ["自然日", "ticker", "交易所委托号", "y_label"]
        exclude_cols = id_cols + leakage_cols
        feature_cols = [col for col in df_train.columns if col not in exclude_cols]
        
        # 移除非数值列
        non_numeric_cols = []
        for col in feature_cols:
            dtype = df_train[col].dtype
            if dtype == 'object' or 'datetime' in str(dtype):
                non_numeric_cols.append(col)
        
        feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
        
        if non_numeric_cols:
            print(f"⚠️ 移除 {len(non_numeric_cols)} 个非数值列: {non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}")
        
        print(f"使用 {len(feature_cols)} 个特征（基础清理）")
    
    X_tr = df_train[feature_cols]
    y_tr = df_train["y_label"]
    X_va = df_valid[feature_cols]
    y_va = df_valid["y_label"]
    
    # Handle missing values
    X_tr = X_tr.fillna(0)
    X_va = X_va.fillna(0)
    
    print(f"Class distribution - Train: {y_tr.value_counts().to_dict()}")
    print(f"Class distribution - Valid: {y_va.value_counts().to_dict()}")
    
    # Advanced sampling (移除SMOTE，使用下采样)
    if args.sampling_method != "none":
        X_tr, y_tr = advanced_sampling(X_tr, y_tr, method=args.sampling_method)
    
    # Model training
    if args.use_ensemble:
        # Train ensemble
        model = EnsembleClassifier()
        model.fit(X_tr, y_tr, X_va, y_va)
    else:
        # Single model with optional hyperparameter optimization
        base_params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'n_estimators': 3000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 10,
            'reg_lambda': 10,
            'random_state': 42,
            'verbose': -1
        }
        
        if args.use_focal_loss:
            print(f"🎯 使用Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
            base_params.update({
                'objective': focal_loss_lgb(args.focal_alpha, args.focal_gamma),
                'metric': 'None'  # 使用自定义objective时需要设置为None
            })
            model = lgb.LGBMClassifier(**base_params)
        elif args.use_class_weight:
            # 计算类别权重
            neg_count = (y_tr == 0).sum()
            pos_count = (y_tr == 1).sum()
            
            if args.class_weight_ratio is not None:
                scale_pos_weight = args.class_weight_ratio
            else:
                scale_pos_weight = neg_count / pos_count
            
            print(f"🎯 使用类别权重 (scale_pos_weight={scale_pos_weight:.2f})")
            print(f"   负样本: {neg_count:,}, 正样本: {pos_count:,}, 比例: {neg_count/pos_count:.1f}:1")
            
            base_params['scale_pos_weight'] = scale_pos_weight
            model = lgb.LGBMClassifier(**base_params)
        elif args.optimize_params:
            best_params = optimize_lgb_params(X_tr, y_tr, X_va, y_va, args.n_trials)
            model = lgb.LGBMClassifier(**best_params, n_estimators=1000, random_state=42, verbose=-1)
        else:
            model = lgb.LGBMClassifier(**base_params)
        
        # 修复early_stopping参数
        if args.use_focal_loss:
            # Focal Loss训练，使用固定轮数，不使用early stopping
            print("📝 Focal Loss模式：使用固定500轮训练（无early stopping）")
            model.n_estimators = 500
            model.fit(X_tr, y_tr)
        else:
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    early_stopping_rounds=100,
                    verbose=False
                )
            except TypeError:
                # 处理新版本LightGBM的兼容性问题
                from lightgbm import early_stopping
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[early_stopping(100)]
                )
    
    # Evaluation
    print("\n📊 Comprehensive Evaluation:")
    if args.use_focal_loss:
        # Focal Loss模式：predict返回1D数组，直接使用
        y_pred_proba = model.predict(X_va)
        # 将logits转换为概率
        y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
    else:
        y_pred_proba = model.predict_proba(X_va)[:, 1]
    
    # 整体评估
    print("\n🌍 Overall Performance:")
    metrics = comprehensive_evaluation(y_va, y_pred_proba)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 分股票评估
    if 'ticker' in df_valid.columns and df_valid['ticker'].nunique() > 1:
        print("\n📈 Per-Ticker Evaluation:")
        ticker_metrics = {}
        
        for ticker in sorted(df_valid['ticker'].unique()):
            ticker_mask = df_valid['ticker'] == ticker
            y_ticker = y_va[ticker_mask]
            pred_ticker = y_pred_proba[ticker_mask]
            
            if len(y_ticker) > 0 and y_ticker.sum() > 0:  # 确保有正样本
                print(f"\n📊 {ticker}:")
                ticker_eval = comprehensive_evaluation(y_ticker, pred_ticker)
                ticker_metrics[ticker] = ticker_eval
                
                # 显示关键指标
                key_metrics = ['PR-AUC', 'ROC-AUC', 'Precision@0.1%', 'Precision@0.5%', 'Precision@1.0%']
                for metric in key_metrics:
                    if metric in ticker_eval:
                        print(f"  {metric}: {ticker_eval[metric]:.6f}")
            else:
                print(f"\n📊 {ticker}: 无正样本或数据不足，跳过评估")
        
        # 股票评估结果汇总表
        if ticker_metrics:
            print(f"\n📋 Per-Ticker Summary:")
            print(f"{'Ticker':<12} {'PR-AUC':<8} {'ROC-AUC':<8} {'P@0.1%':<8} {'P@0.5%':<8} {'P@1.0%':<8} {'Samples':<8} {'Positive':<8}")
            print("-" * 80)
            
            for ticker in sorted(ticker_metrics.keys()):
                tm = ticker_metrics[ticker]
                ticker_mask = df_valid['ticker'] == ticker
                samples = ticker_mask.sum()
                positive = y_va[ticker_mask].sum()
                
                print(f"{ticker:<12} "
                      f"{tm.get('PR-AUC', 0):<8.4f} "
                      f"{tm.get('ROC-AUC', 0):<8.4f} "
                      f"{tm.get('Precision@0.1%', 0):<8.4f} "
                      f"{tm.get('Precision@0.5%', 0):<8.4f} "
                      f"{tm.get('Precision@1.0%', 0):<8.4f} "
                      f"{samples:<8,} "
                      f"{positive:<8,}")
            
            # 保存分股票评估结果
            metrics['per_ticker_metrics'] = ticker_metrics
    else:
        print("\n⚠️ 单股票数据或无股票信息，跳过分股票评估")
    
    # Feature importance (for single models)
    if hasattr(model, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 20 Feature Importance:")
        for i, (feat, imp) in enumerate(feature_imp.head(20).values):
            print(f"  {i+1:2d}. {feat:<30} {imp:>8.0f}")
    
    print(f"\nTotal training time: {time.time()-t0:.1f}s")
    
    # Save comprehensive evaluation results
    method_components = [f"Enhanced+{args.sampling_method}"]
    if args.use_ensemble:
        method_components.append("Ensemble")
    if args.optimize_params:
        method_components.append("Optimized")
    if args.use_focal_loss:
        method_components.append(f"FocalLoss(α={args.focal_alpha},γ={args.focal_gamma})")
    if args.use_class_weight:
        method_components.append(f"ClassWeight({scale_pos_weight:.0f})")
    
    method_name = "+".join(method_components)
    
    # 创建评估结果保存目录
    if args.eval_output_dir is not None:
        eval_output_dir = args.eval_output_dir
    else:
        eval_output_dir = os.path.join("results", "evaluation_results")
    
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"📁 评估结果将保存到: {eval_output_dir}")
    
    # 生成时间戳用于文件命名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存基础结果
    results = {
        'method': method_name,
        'timestamp': timestamp,
        'training_time': time.time()-t0,
        'n_features': len(feature_cols),
        'positive_ratio': y_tr.mean(),
        'train_samples': len(y_tr),
        'valid_samples': len(y_va),
        'train_positive': int(y_tr.sum()),
        'valid_positive': int(y_va.sum()),
        **metrics
    }
    
    # 保存基础结果到JSON
    import json
    results_file = os.path.join(eval_output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 基础评估结果已保存: {results_file}")
    
    # 2. 保存详细的分类报告
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    try:
        # 生成分类报告
        from sklearn.metrics import classification_report, confusion_matrix
        
        class_report = classification_report(y_va, y_pred_binary, output_dict=True)
        
        # 保存分类报告
        class_report_file = os.path.join(eval_output_dir, f"classification_report_{timestamp}.json")
        with open(class_report_file, 'w') as f:
            json.dump(class_report, f, indent=2)
        print(f"✅ 分类报告已保存: {class_report_file}")
        
        # 生成并保存混淆矩阵
        conf_matrix = confusion_matrix(y_va, y_pred_binary)
        
        conf_matrix_data = {
            'confusion_matrix': conf_matrix.tolist(),
            'labels': ['Non-Spoofing', 'Spoofing'],
            'true_negatives': int(conf_matrix[0, 0]),
            'false_positives': int(conf_matrix[0, 1]),
            'false_negatives': int(conf_matrix[1, 0]),
            'true_positives': int(conf_matrix[1, 1])
        }
        
        # 如果有分股票评估，也为每个股票生成混淆矩阵
        if 'per_ticker_metrics' in metrics:
            ticker_confusion_matrices = {}
            for ticker in metrics['per_ticker_metrics'].keys():
                ticker_mask = df_valid['ticker'] == ticker
                y_ticker = y_va[ticker_mask]
                pred_ticker_binary = y_pred_binary[ticker_mask]
                
                if len(y_ticker) > 0:
                    ticker_conf_matrix = confusion_matrix(y_ticker, pred_ticker_binary)
                    ticker_confusion_matrices[ticker] = {
                        'confusion_matrix': ticker_conf_matrix.tolist(),
                        'true_negatives': int(ticker_conf_matrix[0, 0]),
                        'false_positives': int(ticker_conf_matrix[0, 1]),
                        'false_negatives': int(ticker_conf_matrix[1, 0]),
                        'true_positives': int(ticker_conf_matrix[1, 1])
                    }
            
            conf_matrix_data['per_ticker_confusion_matrices'] = ticker_confusion_matrices
        
        conf_matrix_file = os.path.join(eval_output_dir, f"confusion_matrix_{timestamp}.json")
        with open(conf_matrix_file, 'w') as f:
            json.dump(conf_matrix_data, f, indent=2)
        print(f"✅ 混淆矩阵已保存: {conf_matrix_file}")
        
    except Exception as e:
        print(f"⚠️ 分类报告保存失败: {e}")
    
    # 3. 保存预测结果（样本）
    try:
        # 保存预测概率分布
        pred_results = {
            'true_labels': y_va.tolist() if hasattr(y_va, 'tolist') else list(y_va),
            'predicted_probabilities': y_pred_proba.tolist() if hasattr(y_pred_proba, 'tolist') else list(y_pred_proba),
            'predicted_binary': y_pred_binary.tolist() if hasattr(y_pred_binary, 'tolist') else list(y_pred_binary)
        }
        
        # 只保存前10000个样本以节省空间
        if len(pred_results['true_labels']) > 10000:
            pred_results = {
                'true_labels': pred_results['true_labels'][:10000],
                'predicted_probabilities': pred_results['predicted_probabilities'][:10000],
                'predicted_binary': pred_results['predicted_binary'][:10000],
                'note': 'Only first 10,000 samples saved for space efficiency'
            }
        
        pred_results_file = os.path.join(eval_output_dir, f"prediction_results_{timestamp}.json")
        with open(pred_results_file, 'w') as f:
            json.dump(pred_results, f, indent=2)
        print(f"✅ 预测结果已保存: {pred_results_file}")
        
    except Exception as e:
        print(f"⚠️ 预测结果保存失败: {e}")
    
    # 4. 保存评估可视化
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve
        import matplotlib.pyplot as plt
        
        # 创建图表保存目录
        plots_dir = os.path.join(eval_output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # PR曲线
        precision, recall, _ = precision_recall_curve(y_va, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (AUC={metrics["PR-AUC"]:.4f})')
        plt.grid(True)
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_va, y_pred_proba)
        
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, marker='.')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC={metrics["ROC-AUC"]:.4f})')
        plt.grid(True)
        
        plt.tight_layout()
        
        curves_plot_file = os.path.join(plots_dir, f"pr_roc_curves_{timestamp}.png")
        plt.savefig(curves_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ PR/ROC曲线图已保存: {curves_plot_file}")
        
        # 预测概率分布
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(y_pred_proba[y_va == 0], bins=50, alpha=0.7, label='Non-Spoofing', color='blue')
        plt.hist(y_pred_proba[y_va == 1], bins=50, alpha=0.7, label='Spoofing', color='red')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True)
        
        # Precision@K可视化
        k_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        precisions_at_k = []
        theoretical_maxs = []
        
        total_pos = y_va.sum()
        for k in k_values:
            k_int = max(1, int(len(y_va) * k))
            top_k_idx = y_pred_proba.argsort()[::-1][:k_int]
            prec_k = y_va.iloc[top_k_idx].mean() if hasattr(y_va, 'iloc') else y_va[top_k_idx].mean()
            theoretical_max = min(total_pos / k_int, 1.0)
            precisions_at_k.append(prec_k)
            theoretical_maxs.append(theoretical_max)
        
        plt.subplot(1, 3, 2)
        x_pos = range(len(k_values))
        plt.bar([x - 0.2 for x in x_pos], precisions_at_k, 0.4, label='Actual', alpha=0.8)
        plt.bar([x + 0.2 for x in x_pos], theoretical_maxs, 0.4, label='Theoretical Max', alpha=0.8)
        plt.xlabel('Top K%')
        plt.ylabel('Precision@K')
        plt.title('Precision at Different K Values')
        plt.xticks(x_pos, [f'{k*100:.1f}%' for k in k_values])
        plt.legend()
        plt.grid(True)
        
        # 混淆矩阵可视化
        plt.subplot(1, 3, 3)
        import seaborn as sns
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Spoofing', 'Spoofing'],
                   yticklabels=['Non-Spoofing', 'Spoofing'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        
        analysis_plot_file = os.path.join(plots_dir, f"detailed_analysis_{timestamp}.png")
        plt.savefig(analysis_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 详细分析图已保存: {analysis_plot_file}")
        
    except Exception as e:
        print(f"⚠️ 可视化保存失败: {e}")
    
    # 5. 生成评估报告文档
    try:
        report_file = os.path.join(eval_output_dir, f"evaluation_report_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"""# Spoofing Detection Model Evaluation Report

## 实验信息
- **实验时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **模型方法**: {method_name}
- **训练时间**: {results['training_time']:.1f}秒
- **特征数量**: {results['n_features']}

## 数据分布
- **训练集**: {results['train_samples']:,} 样本 ({results['train_positive']:,} 正样本, {results['train_positive']/results['train_samples']*100:.3f}%)
- **验证集**: {results['valid_samples']:,} 样本 ({results['valid_positive']:,} 正样本, {results['valid_positive']/results['valid_samples']*100:.3f}%)

## 核心指标
- **PR-AUC**: {metrics['PR-AUC']:.6f}
- **ROC-AUC**: {metrics['ROC-AUC']:.6f}

## Precision@K 分析
""")
            
            for k in [0.001, 0.005, 0.01, 0.05]:
                if f'Precision@{k*100:.1f}%' in metrics:
                    prec = metrics[f'Precision@{k*100:.1f}%']
                    max_prec = metrics[f'Precision@{k*100:.1f}%_max']
                    achievement = metrics[f'Precision@{k*100:.1f}%_achievement']
                    f.write(f"- **K={k*100:4.1f}%**: {prec:.6f} / {max_prec:.6f} = {achievement*100:5.1f}% 达成率\n")
            
            f.write(f"""
## 混淆矩阵
- **True Negatives**: {conf_matrix_data['true_negatives']:,}
- **False Positives**: {conf_matrix_data['false_positives']:,}
- **False Negatives**: {conf_matrix_data['false_negatives']:,}
- **True Positives**: {conf_matrix_data['true_positives']:,}
""")
            
            # 如果有分股票评估结果，添加到报告中
            if 'per_ticker_metrics' in metrics:
                f.write(f"""
## 分股票评估结果

| 股票代码 | PR-AUC | ROC-AUC | P@0.1% | P@0.5% | P@1.0% | 样本数 | 正样本数 |
|---------|--------|---------|---------|---------|---------|---------|----------|
""")
                for ticker in sorted(metrics['per_ticker_metrics'].keys()):
                    tm = metrics['per_ticker_metrics'][ticker]
                    ticker_mask = df_valid['ticker'] == ticker
                    samples = ticker_mask.sum()
                    positive = y_va[ticker_mask].sum()
                    
                    f.write(f"| {ticker} | {tm.get('PR-AUC', 0):.4f} | "
                           f"{tm.get('ROC-AUC', 0):.4f} | "
                           f"{tm.get('Precision@0.1%', 0):.4f} | "
                           f"{tm.get('Precision@0.5%', 0):.4f} | "
                           f"{tm.get('Precision@1.0%', 0):.4f} | "
                           f"{samples:,} | {positive:,} |\n")
                
                f.write(f"""
### 分股票混淆矩阵

""")
                if 'per_ticker_confusion_matrices' in conf_matrix_data:
                    for ticker, ticker_conf in conf_matrix_data['per_ticker_confusion_matrices'].items():
                        f.write(f"""
**{ticker}:**
- True Negatives: {ticker_conf['true_negatives']:,}
- False Positives: {ticker_conf['false_positives']:,}
- False Negatives: {ticker_conf['false_negatives']:,}
- True Positives: {ticker_conf['true_positives']:,}
""")
            
            f.write(f"""
## 训练参数
```json
{json.dumps(vars(args), indent=2)}
```

## 文件说明
- `evaluation_results_{timestamp}.json`: 基础评估指标
- `classification_report_{timestamp}.json`: 详细分类报告
- `confusion_matrix_{timestamp}.json`: 混淆矩阵数据
- `prediction_results_{timestamp}.json`: 预测结果样本
- `plots/pr_roc_curves_{timestamp}.png`: PR和ROC曲线图
- `plots/detailed_analysis_{timestamp}.png`: 详细分析图表

## 使用建议
基于当前结果，建议：
""")
            
            # 根据结果给出建议
            pr_auc = metrics['PR-AUC']
            if pr_auc >= 0.3:
                f.write("- ✅ 模型性能良好，可以投入使用\n")
            elif pr_auc >= 0.1:
                f.write("- ⚠️ 模型性能中等，建议进一步优化\n")
            else:
                f.write("- ❌ 模型性能较差，需要重新设计特征工程和算法\n")
            
            if metrics.get('Precision@0.1%', 0) >= 0.5:
                f.write("- ✅ Top 0.1% 精度良好，适合高置信度预警\n")
            else:
                f.write("- ⚠️ Top 0.1% 精度不足，建议调整决策阈值\n")
        
        print(f"✅ 评估报告已保存: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 评估报告生成失败: {e}")
    
    # 同时保留原有的简单保存方式（向后兼容）
    simple_results_file = os.path.join(args.data_root, "enhanced_results.json")
    with open(simple_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 兼容性结果文件已保存: {simple_results_file}")
    
    print(f"\n📁 所有评估结果已保存到目录: {eval_output_dir}")
    print(f"📋 查看完整报告: {report_file}")
    
    # Save model and features for analysis
    print("\n💾 保存模型供分析脚本使用...")
    model_output_dir = os.path.join("results", "trained_models")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 确定模型名称
    model_name = f"spoofing_model_{results['method'].replace('+', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace(' ', '_')}"
    
    try:
        # 保存模型和特征
        if args.use_ensemble:
            # 集成模型：保存主要的LightGBM模型
            if hasattr(model, 'fitted_models') and 'lgb' in model.fitted_models:
                main_model = model.fitted_models['lgb']
                model_path = os.path.join(model_output_dir, f"{model_name}.pkl")
                
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': main_model,
                        'ensemble_model': model,  # 完整集成模型
                        'features': feature_cols,
                        'model_type': 'ensemble_lgb',
                        'results': results,
                        'training_params': vars(args)
                    }, f)
                print(f"✅ 集成模型已保存: {model_path}")
            else:
                print("⚠️ 集成模型保存失败：未找到LightGBM子模型")
        else:
            # 单模型
            model_path = os.path.join(model_output_dir, f"{model_name}.pkl")
            
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'features': feature_cols,
                    'model_type': 'lightgbm_single',
                    'results': results,
                    'training_params': vars(args)
                }, f)
            print(f"✅ 单模型已保存: {model_path}")
        
        # 保存特征列表
        features_path = os.path.join(model_output_dir, f"{model_name}_features.json")
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"✅ 特征列表已保存: {features_path}")
        
        # 保存使用说明
        readme_path = os.path.join(model_output_dir, "使用说明.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# 训练好的Spoofing检测模型

## 模型信息
- **模型文件**: `{model_name}.pkl`
- **特征文件**: `{model_name}_features.json`
- **模型类型**: {results['method']}
- **特征数量**: {len(feature_cols)}
- **训练时间**: {results['training_time']:.1f}s
- **PR-AUC**: {results.get('PR-AUC', 'N/A')}
- **ROC-AUC**: {results.get('ROC-AUC', 'N/A')}

## 在操纵时段分析中使用

### 使用pkl格式（推荐）：
```bash
python scripts/analysis/manipulation_detection_heatmap.py \\
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \\
  --model_path "{os.path.relpath(model_path)}" \\
  --output_dir "results/manipulation_analysis"

```

### 如果需要分离的特征文件：
```bash
python scripts/analysis/manipulation_detection_heatmap.py \\
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \\
  --model_path "{os.path.relpath(model_path)}" \\
  --model_features_path "{os.path.relpath(features_path)}" \\
  --output_dir "results/manipulation_analysis"
```

## 训练参数
```json
{json.dumps(vars(args), indent=2)}
```
""")
        print(f"✅ 使用说明已保存: {readme_path}")
        
        # 输出完整的分析命令
        print(f"\n🎯 现在可以运行操纵时段分析:")
        print(f"python scripts/analysis/manipulation_detection_heatmap.py \\")
        print(f"  --data_root \"{args.data_root}\" \\")
        print(f"  --model_path \"{os.path.relpath(model_path)}\" \\")
        print(f"  --output_dir \"results/manipulation_analysis\"")
        
    except Exception as e:
        print(f"⚠️ 模型保存失败: {e}")
        print("模型未保存，但训练结果已记录")

if __name__ == "__main__":
    main() 
    
"""
# 使用明确定义的特征和标签列进行训练
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_ensemble \
  --eval_output_dir "results/my_evaluation_results"

# 指定股票代码进行训练的示例：

# 1. 只训练特定股票
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --include_tickers 000001.SZ 000002.SZ 600000.SH \
  --sampling_method "undersample"

# 2. 排除某些股票
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --exclude_tickers 000001.SZ 000002.SZ \
  --sampling_method "undersample"

# 3. 从文件读取股票列表
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --ticker_file "tickers.txt" \
  --sampling_method "undersample"

# 4. 限制股票数量并按交易量选择
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --max_tickers 10 \
  --ticker_selection_method "by_volume" \
  --sampling_method "undersample"

# 5. 按正样本数量选择股票
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --max_tickers 5 \
  --ticker_selection_method "by_positive_samples" \
  --use_focal_loss

# 列定义模块使用说明:
# 特征列定义在: scripts/data_process/features/feature_generator.py
# 标签列定义在: scripts/data_process/labels/label_generator.py
# 
# 主要函数：
# - get_training_feature_columns(): 获取可用于训练的特征列
# - get_training_target_columns(): 获取可用作目标变量的标签列
# - get_leakage_risk_columns(): 获取有数据泄露风险的列

# 股票筛选参数说明：
# --include_tickers: 指定包含的股票代码列表
# --exclude_tickers: 指定排除的股票代码列表  
# --ticker_file: 从文件读取股票代码列表（每行一个）
# --max_tickers: 最大股票数量限制
# --ticker_selection_method: 股票选择方法（random/by_volume/by_positive_samples）
"""