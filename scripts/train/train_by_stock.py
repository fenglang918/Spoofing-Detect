#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock-wise LightGBM Training Pipeline for Spoofing Detection
------------------------------------------------------------
主要特点：
• 分股票进行训练和评估
• 支持模型集成策略
• 增强特征工程
• 详细的分股票性能分析
• 支持跨股票泛化测试
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
from collections import defaultdict
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ---------- 导入现有的工具函数 --------------------------------
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
    print(f"🔧 Creating enhanced features for {len(df)} samples...")
    
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
        tech_features = tech_features.loc[:, ~tech_features.columns.isin(df.columns)]
        if not tech_features.empty:
            feature_dfs.append(tech_features)
    if not stat_features.empty:
        stat_features = stat_features.loc[:, ~stat_features.columns.isin(df.columns)]
        if not stat_features.empty:
            feature_dfs.append(stat_features)
    if not interaction_features.empty:
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
    
    return enhanced_df

def advanced_sampling(X, y, method='undersample', sampling_strategy='auto'):
    """高级采样策略"""
    if method == 'none':
        return X, y
    
    print(f"Before sampling: {X.shape[0]} samples, {y.sum()} positive ({y.mean():.3%})")
    
    if method == 'undersample':
        # 简单下采样到1:10比例
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        n_pos = len(pos_indices)
        n_neg_target = min(n_pos * 10, len(neg_indices))  # 1:10比例
        
        if n_neg_target < len(neg_indices):
            selected_neg = np.random.choice(neg_indices, n_neg_target, replace=False)
            selected_indices = np.concatenate([pos_indices, selected_neg])
            X_resampled = X.iloc[selected_indices]
            y_resampled = y.iloc[selected_indices]
        else:
            X_resampled, y_resampled = X, y
    
    elif method == 'stratified_undersample':
        # 分层下采样（如果有ticker信息）
        if hasattr(X, 'ticker') or 'ticker' in X.columns:
            X_resampled_list = []
            y_resampled_list = []
            
            for ticker in X['ticker'].unique():
                ticker_mask = X['ticker'] == ticker
                X_ticker = X[ticker_mask]
                y_ticker = y[ticker_mask]
                
                if y_ticker.sum() > 0:  # 只对有正样本的股票进行采样
                    X_ticker_sampled, y_ticker_sampled = advanced_sampling(
                        X_ticker, y_ticker, method='undersample'
                    )
                    X_resampled_list.append(X_ticker_sampled)
                    y_resampled_list.append(y_ticker_sampled)
            
            if X_resampled_list:
                X_resampled = pd.concat(X_resampled_list, ignore_index=True)
                y_resampled = pd.concat(y_resampled_list, ignore_index=True)
            else:
                X_resampled, y_resampled = X, y
        else:
            # 回退到普通下采样
            X_resampled, y_resampled = advanced_sampling(X, y, method='undersample')
    
    else:
        X_resampled, y_resampled = X, y
    
    print(f"After sampling: {X_resampled.shape[0]} samples, {y_resampled.sum()} positive ({y_resampled.mean():.3%})")
    return X_resampled, y_resampled

class EnsembleClassifier:
    """集成分类器"""
    def __init__(self, models=None):
        self.models = models or ['lgb', 'xgb', 'rf']
        self.fitted_models = {}
        self.weights = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        """训练所有子模型"""
        print("🔧 Training ensemble models...")
        
        # LightGBM
        if 'lgb' in self.models:
            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                metric='average_precision',
                learning_rate=0.03,
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
            
            if X_val is not None and y_val is not None:
                try:
                    lgb_model.fit(
                        X, y,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=100,
                        verbose=False
                    )
                except TypeError:
                    from lightgbm import early_stopping
                    lgb_model.fit(
                        X, y,
                        eval_set=[(X_val, y_val)],
                        callbacks=[early_stopping(100)]
                    )
            else:
                lgb_model.fit(X, y)
            
            self.fitted_models['lgb'] = lgb_model
            print("✅ LightGBM trained")
        
        # XGBoost
        if 'xgb' in self.models:
            xgb_model = XGBClassifier(
                learning_rate=0.03,
                max_depth=6,
                n_estimators=1000,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=10,
                reg_lambda=10,
                random_state=42,
                eval_metric='aucpr',
                verbosity=0
            )
            
            if X_val is not None and y_val is not None:
                xgb_model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=100,
                    verbose=False
                )
            else:
                xgb_model.fit(X, y)
            
            self.fitted_models['xgb'] = xgb_model
            print("✅ XGBoost trained")
        
        # Random Forest
        if 'rf' in self.models:
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X, y)
            self.fitted_models['rf'] = rf_model
            print("✅ Random Forest trained")
        
        # 计算权重
        if X_val is not None and y_val is not None:
            self._compute_weights(X_val, y_val)
    
    def _compute_weights(self, X_val, y_val):
        """基于验证集计算模型权重"""
        scores = {}
        for name, model in self.fitted_models.items():
            y_pred = model.predict_proba(X_val)[:, 1]
            scores[name] = average_precision_score(y_val, y_pred)
        
        total_score = sum(scores.values())
        self.weights = {name: score/total_score for name, score in scores.items()}
        print(f"Model weights: {self.weights}")
    
    def predict_proba(self, X):
        """集成预测"""
        if not self.fitted_models:
            raise ValueError("No models have been fitted")
        
        predictions = []
        weights = []
        
        for name, model in self.fitted_models.items():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
            weights.append(self.weights.get(name, 1.0) if self.weights else 1.0)
        
        # 加权平均
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        # 返回格式与sklearn兼容
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

def comprehensive_evaluation(y_true, y_pred_proba, y_pred_binary=None):
    """综合评估"""
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # 基础指标
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    metrics = {
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc
    }
    
    # Precision@K
    total_positive = y_true.sum()
    n_samples = len(y_true)
    
    for k in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        k_samples = max(1, int(n_samples * k))
        top_k_indices = np.argsort(y_pred_proba)[::-1][:k_samples]
        
        precision_at_k = y_true.iloc[top_k_indices].mean() if hasattr(y_true, 'iloc') else y_true[top_k_indices].mean()
        theoretical_max = min(total_positive / k_samples, 1.0)
        achievement_rate = precision_at_k / theoretical_max if theoretical_max > 0 else 0
        
        metrics[f'Precision@{k*100:.1f}%'] = precision_at_k
        metrics[f'Precision@{k*100:.1f}%_max'] = theoretical_max
        metrics[f'Precision@{k*100:.1f}%_achievement'] = achievement_rate
    
    return metrics

def train_stock_model(df_stock, feature_cols, args, stock_name):
    """为单只股票训练模型"""
    print(f"\n📈 Training model for {stock_name}...")
    
    # 分割数据
    train_mask = df_stock["自然日"].astype(str).str.contains(args.train_regex)
    valid_mask = df_stock["自然日"].astype(str).str.contains(args.valid_regex)
    
    df_train = df_stock[train_mask].copy()
    df_valid = df_stock[valid_mask].copy()
    
    if len(df_train) == 0 or len(df_valid) == 0:
        print(f"❌ {stock_name}: Insufficient data (train={len(df_train)}, valid={len(df_valid)})")
        return None
    
    if df_train['y_label'].sum() == 0:
        print(f"❌ {stock_name}: No positive samples in training data")
        return None
    
    print(f"   Training: {len(df_train):,}, Validation: {len(df_valid):,}")
    print(f"   Train positive: {df_train['y_label'].sum()}/{len(df_train)} ({df_train['y_label'].mean():.3%})")
    print(f"   Valid positive: {df_valid['y_label'].sum()}/{len(df_valid)} ({df_valid['y_label'].mean():.3%})")
    
    # 增强特征工程
    df_train = enhance_features(df_train)
    df_valid = enhance_features(df_valid)
    
    # 特征选择
    available_features = [col for col in feature_cols if col in df_train.columns]
    if len(available_features) < len(feature_cols):
        missing = [col for col in feature_cols if col not in df_train.columns]
        print(f"   ⚠️ Missing {len(missing)} features: {missing[:3]}...")
    
    X_tr = df_train[available_features].fillna(0)
    y_tr = df_train["y_label"]
    X_va = df_valid[available_features].fillna(0)
    y_va = df_valid["y_label"]
    
    # 采样
    if args.sampling_method != "none":
        X_tr, y_tr = advanced_sampling(X_tr, y_tr, method=args.sampling_method)
    
    # 训练模型
    start_time = time.time()
    
    if args.use_ensemble:
        model = EnsembleClassifier()
        model.fit(X_tr, y_tr, X_va, y_va)
    else:
        # 单模型
        base_params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 10,
            'reg_lambda': 10,
            'random_state': 42,
            'verbose': -1
        }
        
        if args.use_focal_loss:
            base_params.update({
                'objective': focal_loss_lgb(args.focal_alpha, args.focal_gamma),
                'metric': 'None'
            })
            model = lgb.LGBMClassifier(**base_params)
            model.n_estimators = 500
            model.fit(X_tr, y_tr)
        elif args.use_class_weight:
            neg_count = (y_tr == 0).sum()
            pos_count = (y_tr == 1).sum()
            
            if args.class_weight_ratio is not None:
                scale_pos_weight = args.class_weight_ratio
            else:
                scale_pos_weight = neg_count / pos_count
            
            base_params['scale_pos_weight'] = scale_pos_weight
            model = lgb.LGBMClassifier(**base_params)
        else:
            model = lgb.LGBMClassifier(**base_params)
        
        # 训练
        if not args.use_focal_loss:
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    early_stopping_rounds=100,
                    verbose=False
                )
            except TypeError:
                from lightgbm import early_stopping
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[early_stopping(100)]
                )
    
    training_time = time.time() - start_time
    
    # 预测
    if args.use_focal_loss and not args.use_ensemble:
        y_pred_proba = model.predict(X_va)
        y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
    else:
        y_pred_proba = model.predict_proba(X_va)[:, 1]
    
    # 评估
    metrics = comprehensive_evaluation(y_va, y_pred_proba)
    
    result = {
        'stock': stock_name,
        'model': model,
        'features': available_features,
        'training_time': training_time,
        'train_samples': len(y_tr),
        'valid_samples': len(y_va),
        'train_positive': int(y_tr.sum()),
        'valid_positive': int(y_va.sum()),
        'metrics': metrics,
        'y_true': y_va,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"   ✅ {stock_name}: PR-AUC={metrics['PR-AUC']:.4f}, P@0.1%={metrics.get('Precision@0.1%', 0):.4f}, Time={training_time:.1f}s")
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_regex", default="202503|202504")
    parser.add_argument("--valid_regex", default="202505")
    parser.add_argument("--sampling_method", choices=["none", "undersample", "stratified_undersample"], 
                       default="undersample", help="采样方法")
    parser.add_argument("--use_ensemble", action="store_true", help="使用模型集成")
    parser.add_argument("--use_enhanced_labels", action="store_true", 
                       help="使用增强标签而不是原始y_label")
    parser.add_argument("--label_type", choices=["composite_spoofing", "conservative_spoofing"], 
                       default="composite_spoofing", help="使用哪种增强标签类型")
    parser.add_argument("--use_focal_loss", action="store_true", help="使用Focal Loss处理不平衡数据")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal Loss alpha参数")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma参数")
    parser.add_argument("--use_class_weight", action="store_true", help="使用类别权重处理不平衡数据")
    parser.add_argument("--class_weight_ratio", type=float, default=None, help="正样本权重比例")
    parser.add_argument("--min_samples", type=int, default=1000, help="股票最小样本数")
    parser.add_argument("--min_positive", type=int, default=10, help="股票最小正样本数")
    parser.add_argument("--eval_output_dir", type=str, default=None, 
                       help="评估结果保存目录，默认为 results/stock_wise_results")
    
    args = parser.parse_args()
    
    t0 = time.time()
    print("🔍 Loading and preparing data...")
    
    # Load data
    feat_pats = [os.path.join(args.data_root, "features", "X_*.parquet")]
    lab_pats = [os.path.join(args.data_root, "labels", "labels_*.parquet")]
    
    # 如果新路径不存在，回退到旧路径
    if not glob.glob(feat_pats[0]):
        print("⚠️ New architecture paths not found, trying legacy paths...")
        feat_pats = [os.path.join(args.data_root, "features_select", "X_*.parquet")]
        lab_pats = [os.path.join(args.data_root, "labels_select", "labels_*.parquet")]
    
    # 加载特征数据
    files = []
    for pat in feat_pats:
        files.extend(sorted(glob.glob(pat)))
    
    if not files:
        print(f"❌ No feature files found.")
        return
    
    print(f"📊 Loading features from {len(files)} files...")
    df_feat = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    # 加载标签数据
    files = []
    for pat in lab_pats:
        files.extend(sorted(glob.glob(pat)))
    
    if not files:
        enhanced_lab_pat = os.path.join(args.data_root, "labels_enhanced", "labels_*.parquet")
        files.extend(sorted(glob.glob(enhanced_lab_pat)))
    
    if not files:
        print(f"❌ No label files found.")
        return
    
    print(f"📊 Loading labels from {len(files)} files...")
    df_lab = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    # 检查增强标签
    if args.use_enhanced_labels:
        if args.label_type not in df_lab.columns:
            print(f"❌ Enhanced label '{args.label_type}' not found in labels.")
            return
        print(f"✅ Using enhanced label: {args.label_type}")
    
    # 合并数据
    df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="inner")
    print(f"Merged data shape: {df.shape}")
    
    # 设置目标变量
    if args.use_enhanced_labels:
        df['y_label'] = df[args.label_type]
    
    # 获取特征列
    leakage_cols = [
        "存活时间_ms", "事件_datetime", "成交价格", "成交数量", "事件类型",
        "is_cancel_event", "is_trade_event",
        "flag_R1", "flag_R2", "enhanced_spoofing_liberal", 
        "enhanced_spoofing_moderate", "enhanced_spoofing_strict"
    ]
    
    id_cols = ["自然日", "ticker", "交易所委托号", "y_label"]
    exclude_cols = id_cols + leakage_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 移除非数值列
    non_numeric_cols = []
    for col in feature_cols:
        dtype = df[col].dtype
        if dtype == 'object' or 'datetime' in str(dtype):
            non_numeric_cols.append(col)
    
    feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
    print(f"Using {len(feature_cols)} features")
    
    # 按股票分组处理
    print(f"\n📈 Analyzing stocks...")
    stock_stats = df.groupby('ticker').agg({
        'y_label': ['count', 'sum', 'mean']
    }).round(4)
    stock_stats.columns = ['total_samples', 'positive_samples', 'positive_rate']
    stock_stats = stock_stats.sort_values('positive_samples', ascending=False)
    
    print(f"Stock statistics:")
    print(stock_stats.head(10))
    
    # 筛选符合条件的股票
    eligible_stocks = stock_stats[
        (stock_stats['total_samples'] >= args.min_samples) & 
        (stock_stats['positive_samples'] >= args.min_positive)
    ].index.tolist()
    
    print(f"\n✅ Found {len(eligible_stocks)} eligible stocks (min_samples={args.min_samples}, min_positive={args.min_positive})")
    print(f"Eligible stocks: {eligible_stocks}")
    
    if len(eligible_stocks) == 0:
        print("❌ No stocks meet the minimum requirements")
        return
    
    # 为每只股票训练模型
    print(f"\n🚀 Training models for {len(eligible_stocks)} stocks...")
    stock_results = []
    
    for stock in eligible_stocks:
        df_stock = df[df['ticker'] == stock].copy()
        result = train_stock_model(df_stock, feature_cols, args, stock)
        if result:
            stock_results.append(result)
    
    if not stock_results:
        print("❌ No successful stock models trained")
        return
    
    print(f"\n✅ Successfully trained {len(stock_results)} stock models")
    
    # 汇总结果分析
    print(f"\n📊 Stock-wise Performance Analysis:")
    print("=" * 80)
    
    # 创建结果表格
    results_summary = []
    for result in stock_results:
        summary = {
            'Stock': result['stock'],
            'TrainSamples': result['train_samples'],
            'ValidSamples': result['valid_samples'],
            'TrainPos': result['train_positive'],
            'ValidPos': result['valid_positive'],
            'PR-AUC': result['metrics']['PR-AUC'],
            'ROC-AUC': result['metrics']['ROC-AUC'],
            'P@0.1%': result['metrics'].get('Precision@0.1%', 0),
            'P@1%': result['metrics'].get('Precision@1.0%', 0),
            'TrainingTime': result['training_time']
        }
        results_summary.append(summary)
    
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('PR-AUC', ascending=False)
    
    print(results_df.round(4).to_string(index=False))
    
    # 统计分析
    print(f"\n📈 Performance Statistics:")
    print(f"Average PR-AUC: {results_df['PR-AUC'].mean():.4f} ± {results_df['PR-AUC'].std():.4f}")
    print(f"Best PR-AUC: {results_df['PR-AUC'].max():.4f} ({results_df.loc[results_df['PR-AUC'].idxmax(), 'Stock']})")
    print(f"Worst PR-AUC: {results_df['PR-AUC'].min():.4f} ({results_df.loc[results_df['PR-AUC'].idxmin(), 'Stock']})")
    print(f"Total training time: {results_df['TrainingTime'].sum():.1f}s")
    
    # 跨股票泛化测试（可选）
    print(f"\n🔄 Cross-stock Generalization Test...")
    cross_stock_results = []
    
    for i, train_result in enumerate(stock_results[:3]):  # 只测试前3个表现最好的股票
        train_stock = train_result['stock']
        train_model = train_result['model']
        
        for j, test_result in enumerate(stock_results):
            test_stock = test_result['stock']
            if train_stock == test_stock:
                continue
            
            # 使用训练股票的模型预测测试股票
            try:
                if args.use_focal_loss and not args.use_ensemble:
                    y_pred_cross = train_model.predict(test_result['y_true'].values.reshape(-1, len(train_result['features'])))
                    y_pred_cross = 1 / (1 + np.exp(-y_pred_cross))
                else:
                    # 需要构造测试数据
                    df_test_stock = df[df['ticker'] == test_stock].copy()
                    test_mask = df_test_stock["自然日"].astype(str).str.contains(args.valid_regex)
                    df_test = df_test_stock[test_mask].copy()
                    
                    if len(df_test) == 0:
                        continue
                    
                    df_test = enhance_features(df_test)
                    available_features = [col for col in train_result['features'] if col in df_test.columns]
                    X_test = df_test[available_features].fillna(0)
                    
                    if len(available_features) < len(train_result['features']) * 0.8:  # 如果特征缺失太多，跳过
                        continue
                    
                    y_pred_cross = train_model.predict_proba(X_test)[:, 1]
                
                cross_metrics = comprehensive_evaluation(test_result['y_true'], y_pred_cross)
                cross_stock_results.append({
                    'TrainStock': train_stock,
                    'TestStock': test_stock,
                    'CrossPR-AUC': cross_metrics['PR-AUC'],
                    'OriginalPR-AUC': test_result['metrics']['PR-AUC'],
                    'Performance_Ratio': cross_metrics['PR-AUC'] / test_result['metrics']['PR-AUC']
                })
                
            except Exception as e:
                print(f"⚠️ Cross-stock test failed ({train_stock}->{test_stock}): {e}")
                continue
    
    if cross_stock_results:
        cross_df = pd.DataFrame(cross_stock_results)
        print("\nCross-stock Performance (Top examples):")
        print(cross_df.head(10).round(4).to_string(index=False))
        print(f"\nAverage cross-stock performance ratio: {cross_df['Performance_Ratio'].mean():.3f}")
    
    # 保存结果
    if args.eval_output_dir is not None:
        eval_output_dir = args.eval_output_dir
    else:
        eval_output_dir = os.path.join("results", "stock_wise_results")
    
    os.makedirs(eval_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    detailed_results = {
        'summary': results_summary,
        'cross_stock': cross_stock_results,
        'args': vars(args),
        'timestamp': timestamp,
        'total_time': time.time() - t0
    }
    
    results_file = os.path.join(eval_output_dir, f"stock_wise_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    # 保存CSV表格
    csv_file = os.path.join(eval_output_dir, f"stock_performance_{timestamp}.csv")
    results_df.to_csv(csv_file, index=False)
    
    if cross_stock_results:
        cross_csv_file = os.path.join(eval_output_dir, f"cross_stock_performance_{timestamp}.csv")
        cross_df.to_csv(cross_csv_file, index=False)
    
    print(f"\n📁 Results saved to: {eval_output_dir}")
    print(f"📊 Performance table: {csv_file}")
    print(f"📋 Detailed results: {results_file}")
    
    print(f"\nTotal execution time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()