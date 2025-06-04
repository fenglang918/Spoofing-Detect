#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced LightGBM Training Pipeline for Spoofing Detection
----------------------------------------------------------
主要优化：
• 增强特征工程：技术指标、统计特征、交互特征
• 改进数据平衡策略：SMOTE + 分层采样
• 模型集成：多种算法组合
• 更细致的超参数调优
• 增强评估指标和可视化
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
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ---------- Enhanced Feature Engineering --------------------------------
def create_technical_indicators(df):
    """创建技术指标特征"""
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
    features['price_improvement'] = np.where(
        df['is_buy'] == 1,
        (df['委托价格'] - df['ask1']) / df['ask1'],
        (df['bid1'] - df['委托价格']) / df['bid1']
    )
    
    # 时间序列特征
    df_sorted = df.sort_values(['ticker', '委托_datetime'])
    features['order_rate_1min'] = df_sorted.groupby('ticker')['委托_datetime'].transform(
        lambda x: x.rolling('1min').count()
    )
    features['cancel_rate_1min'] = df_sorted.groupby('ticker')['is_cancel'].transform(
        lambda x: x.rolling('1min').sum()
    )
    
    # 价格级别特征
    features['at_bid'] = (df['委托价格'] <= df['bid1']).astype(int)
    features['at_ask'] = (df['委托价格'] >= df['ask1']).astype(int)
    features['between_quotes'] = ((df['委托价格'] > df['bid1']) & 
                                 (df['委托价格'] < df['ask1'])).astype(int)
    
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
    
    # 历史行为特征
    df_sorted = df.sort_values(['ticker', '委托_datetime'])
    features['hist_order_size_mean'] = df_sorted.groupby('ticker')['log_qty'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    features['hist_spread_mean'] = df_sorted.groupby('ticker')['spread'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    return pd.DataFrame(features)

def create_interaction_features(df):
    """创建交互特征"""
    features = {}
    
    # 关键交互
    features['qty_spread_interaction'] = df['log_qty'] * df['relative_spread']
    features['time_qty_interaction'] = df['time_cos'] * df['log_qty']
    features['direction_spread_interaction'] = df['is_buy'] * df['pct_spread']
    
    # 价格-时间交互
    features['price_time_interaction'] = df['price_dev_prevclose'] * df['time_sin']
    
    return pd.DataFrame(features)

def enhance_features(df):
    """增强特征工程"""
    print("🔧 Creating enhanced features...")
    
    # 计算基础衍生特征
    df['relative_spread'] = df['spread'] / df['mid_price']
    
    # 创建技术指标
    tech_features = create_technical_indicators(df)
    
    # 创建统计特征
    stat_features = create_statistical_features(df)
    
    # 创建交互特征
    interaction_features = create_interaction_features(df)
    
    # 合并所有特征
    enhanced_df = pd.concat([df, tech_features, stat_features, interaction_features], axis=1)
    
    # 处理无穷值和缺失值
    enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
    enhanced_df = enhanced_df.fillna(0)
    
    return enhanced_df

# ---------- Advanced Sampling Strategies --------------------------------
def advanced_sampling(X, y, method='smote_tomek', sampling_strategy='auto'):
    """高级采样策略"""
    print(f"🎯 Applying {method} sampling...")
    
    original_pos = y.sum()
    original_neg = len(y) - original_pos
    print(f"Original distribution: {original_pos:,} positive, {original_neg:,} negative")
    
    if method == 'smote':
        sampler = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=42, sampling_strategy=sampling_strategy)
    else:
        # 使用原有的下采样
        return X, y
    
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        new_pos = y_resampled.sum()
        new_neg = len(y_resampled) - new_pos
        print(f"After {method}: {new_pos:,} positive, {new_neg:,} negative")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"⚠️ Sampling failed: {e}, using original data")
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
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=10,
                    reg_lambda=10,
                    random_state=42,
                    verbose=0
                ),
                'rf': RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
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
            
            if name in ['lgb', 'xgb'] and X_val is not None:
                model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=100,
                    verbose=False
                )
            else:
                model.fit(X, y)
            
            self.fitted_models[name] = model
        
        # 如果有验证集，计算权重
        if X_val is not None and y_val is not None:
            self._compute_weights(X_val, y_val)
        else:
            # 等权重
            self.weights = {name: 1.0 for name in self.models.keys()}
    
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
        self.weights = {name: score/total_score for name, score in scores.items()}
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
    """使用Optuna优化LightGBM参数"""
    print("🔍 Optimizing hyperparameters...")
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params, n_estimators=1000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
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

def plot_evaluation_results(y_true, y_pred_proba, save_dir=None):
    """绘制评估结果"""
    from sklearn.metrics import precision_recall_curve, roc_curve
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    axes[0, 0].plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Score Distribution
    pos_scores = y_pred_proba[y_true == 1]
    neg_scores = y_pred_proba[y_true == 0]
    axes[1, 0].hist(neg_scores, bins=50, alpha=0.7, label='Negative', density=True)
    axes[1, 0].hist(pos_scores, bins=50, alpha=0.7, label='Positive', density=True)
    axes[1, 0].set_xlabel('Prediction Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Threshold Analysis
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds:
        pred_binary = (y_pred_proba >= thresh).astype(int)
        if pred_binary.sum() > 0:
            prec = (pred_binary & y_true).sum() / pred_binary.sum()
            rec = (pred_binary & y_true).sum() / y_true.sum()
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        else:
            prec = rec = f1 = 0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    axes[1, 1].plot(thresholds, precisions, label='Precision')
    axes[1, 1].plot(thresholds, recalls, label='Recall')
    axes[1, 1].plot(thresholds, f1s, label='F1-Score')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Threshold Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

# ---------- Main Function ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_regex", default="202503|202504")
    parser.add_argument("--valid_regex", default="202505")
    parser.add_argument("--sampling_method", choices=["none", "undersample", "smote", "smote_tomek"], 
                       default="smote_tomek")
    parser.add_argument("--use_ensemble", action="store_true", help="使用模型集成")
    parser.add_argument("--optimize_params", action="store_true", help="优化超参数")
    parser.add_argument("--n_trials", type=int, default=50, help="超参数优化试验次数")
    
    args = parser.parse_args()
    
    t0 = time.time()
    print("🔍 Loading and preparing data...")
    
    # Load data (same as before)
    feat_pats = [os.path.join(args.data_root, "features_select", "X_*.parquet")]
    lab_pats = [os.path.join(args.data_root, "labels_select", "labels_*.parquet")]
    
    files = []
    for pat in feat_pats:
        files.extend(sorted(glob.glob(pat)))
    df_feat = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    files = []
    for pat in lab_pats:
        files.extend(sorted(glob.glob(pat)))
    df_lab = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="inner")
    print(f"Merged data shape: {df.shape}")
    
    # Split data
    train_mask = df["自然日"].astype(str).str.contains(args.train_regex)
    valid_mask = df["自然日"].astype(str).str.contains(args.valid_regex)
    
    df_train = df[train_mask]
    df_valid = df[valid_mask]
    
    print(f"Training: {len(df_train):,}, Validation: {len(df_valid):,}")
    
    # Enhanced feature engineering
    df_train = enhance_features(df_train)
    df_valid = enhance_features(df_valid)
    
    # Prepare features
    id_cols = ["自然日", "ticker", "交易所委托号", "y_label"]
    feature_cols = [col for col in df_train.columns if col not in id_cols]
    
    # Remove any remaining problematic features
    feature_cols = [col for col in feature_cols if not col.startswith('Unnamed')]
    
    print(f"Using {len(feature_cols)} features")
    
    X_tr = df_train[feature_cols]
    y_tr = df_train["y_label"]
    X_va = df_valid[feature_cols]
    y_va = df_valid["y_label"]
    
    # Handle missing values
    X_tr = X_tr.fillna(0)
    X_va = X_va.fillna(0)
    
    print(f"Class distribution - Train: {y_tr.value_counts().to_dict()}")
    print(f"Class distribution - Valid: {y_va.value_counts().to_dict()}")
    
    # Advanced sampling
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
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=100,
        )
    
    # Evaluation
    print("\n📊 Comprehensive Evaluation:")
    y_pred_proba = model.predict_proba(X_va)[:, 1]
    
    metrics = comprehensive_evaluation(y_va, y_pred_proba)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Visualizations
    plot_evaluation_results(y_va, y_pred_proba, args.data_root)
    
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

if __name__ == "__main__":
    main() 