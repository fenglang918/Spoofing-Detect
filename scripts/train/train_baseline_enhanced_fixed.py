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

# ---------- Enhanced Evaluation ------------------------------------------
def comprehensive_evaluation(y_true, y_pred_proba, y_pred_binary=None):
    """综合评估"""
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # 基础指标
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Precision at K
    metrics = {'PR-AUC': pr_auc, 'ROC-AUC': roc_auc}
    
    for k in [0.001, 0.005, 0.01, 0.05]:
        k_int = max(1, int(len(y_true) * k))
        top_k_idx = y_pred_proba.argsort()[::-1][:k_int]
        prec_k = y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()
        metrics[f'Precision@{k*100:.1f}%'] = prec_k
    
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
    
    # Split data
    train_mask = df["自然日"].astype(str).str.contains(args.train_regex)
    valid_mask = df["自然日"].astype(str).str.contains(args.valid_regex)
    
    df_train = df[train_mask].copy()
    df_valid = df[valid_mask].copy()
    
    print(f"Training: {len(df_train):,}, Validation: {len(df_valid):,}")
    
    # Enhanced feature engineering
    df_train = enhance_features(df_train)
    df_valid = enhance_features(df_valid)
    
    # Prepare features
    id_cols = ["自然日", "ticker", "交易所委托号", "y_label"]
    feature_cols = [col for col in df_train.columns if col not in id_cols]
    
    # Remove any remaining problematic features
    feature_cols = [col for col in feature_cols if not col.startswith('Unnamed')]
    
    # Filter out non-numeric columns that LightGBM can't handle
    non_numeric_cols = []
    for col in feature_cols:
        dtype = df_train[col].dtype
        if dtype == 'object' or 'datetime' in str(dtype):
            non_numeric_cols.append(col)
    
    feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
    
    if non_numeric_cols:
        print(f"⚠️ Removed {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols[:10]}{'...' if len(non_numeric_cols) > 10 else ''}")
    
    print(f"Using {len(feature_cols)} numeric features")
    
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
        if args.optimize_params:
            best_params = optimize_lgb_params(X_tr, y_tr, X_va, y_va, args.n_trials)
            model = lgb.LGBMClassifier(**best_params, n_estimators=1000, random_state=42, verbose=-1)
        else:
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
        
        # 修复early_stopping参数
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
    y_pred_proba = model.predict_proba(X_va)[:, 1]
    
    metrics = comprehensive_evaluation(y_va, y_pred_proba)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
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
    
    # Save results
    results = {
        'method': f"Enhanced+{args.sampling_method}" + ("+Ensemble" if args.use_ensemble else "") + ("+Optimized" if args.optimize_params else ""),
        'training_time': time.time()-t0,
        'n_features': len(feature_cols),
        'positive_ratio': y_tr.mean(),
        **metrics
    }
    
    # Save to JSON
    import json
    results_file = os.path.join(args.data_root, "enhanced_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main() 
    
"""
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "undersample" \
  --use_ensemble \
  --optimize_params \
  --n_trials 50
"""