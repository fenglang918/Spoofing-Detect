#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Prediction and Visualization Analysis Script
==============================
Features:
1. Load trained model
2. Load validation set features and labels
3. Make predictions on validation set
4. Plot market data with real and predicted anomaly periods
5. Link to order data for accurate timestamps
"""

import argparse
import glob
import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from pathlib import Path
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# 添加训练脚本路径以便加载模型类和特征函数
sys.path.append(str(Path(__file__).parent.parent / "train"))

# Set matplotlib font for English display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 模型类定义（从训练脚本复制）
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class EnsembleClassifier:
    """集成分类器类（从训练脚本复制）"""
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
                )
            }
            
            if XGBClassifier is not None:
                self.models['xgb'] = XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='aucpr',
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=10,
                    reg_lambda=10,
                    random_state=42,
                    verbosity=0
                )
            
            self.models['rf'] = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
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
                model.fit(X, y)
            
            self.fitted_models[name] = model
        
        if X_val is not None and y_val is not None:
            self._compute_weights(X_val, y_val)
        else:
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
    
    def _compute_weights(self, X_val, y_val):
        """基于验证集性能计算权重"""
        scores = {}
        for name, model in self.fitted_models.items():
            pred_proba = model.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, pred_proba)
            scores[name] = score
            print(f"  {name} PR-AUC: {score:.4f}")
        
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


def load_model(model_path):
    """加载已训练的模型"""
    print(f"📥 Loading model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    features = model_data['features']
    model_type = model_data.get('model_type', 'unknown')
    results = model_data.get('results', {})
    
    print(f"✅ Model loaded successfully")
    print(f"   Type: {model_type}")
    print(f"   Features: {len(features)}")
    print(f"   Performance: PR-AUC={results.get('PR-AUC', 'N/A'):.4f}")
    
    # 如果是集成模型，提取主要模型
    if hasattr(model, 'fitted_models') and 'lgb' in model.fitted_models:
        print("   Using LightGBM from ensemble")
        main_model = model.fitted_models['lgb']
        return main_model, features, model_data
    
    return model, features, model_data


def load_validation_data(data_root, valid_regex):
    """Load validation set features and labels"""
    print(f"📊 Loading validation data with regex: {valid_regex}")
    
    # Load feature data
    feat_pats = [
        os.path.join(data_root, "features", "X_*.parquet"),
        os.path.join(data_root, "features_select", "X_*.parquet")
    ]
    
    feat_files = []
    for pat in feat_pats:
        feat_files.extend(sorted(glob.glob(pat)))
        if feat_files:
            break
    
    if not feat_files:
        raise FileNotFoundError("No feature files found")
    
    print(f"Found {len(feat_files)} feature files")
    df_feat = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    
    # Load label data
    lab_pats = [
        os.path.join(data_root, "labels", "labels_*.parquet"),
        os.path.join(data_root, "labels_select", "labels_*.parquet"),
        os.path.join(data_root, "labels_enhanced", "labels_*.parquet")
    ]
    
    lab_files = []
    for pat in lab_pats:
        lab_files.extend(sorted(glob.glob(pat)))
        if lab_files:
            break
    
    if not lab_files:
        raise FileNotFoundError("No label files found")
    
    print(f"Found {len(lab_files)} label files")
    df_lab = pd.concat([pd.read_parquet(f) for f in lab_files], ignore_index=True)
    
    # Merge data
    df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="inner")
    print(f"Merged data shape: {df.shape}")
    
    # Filter validation set
    valid_mask = df["自然日"].astype(str).str.contains(valid_regex)
    df_valid = df[valid_mask].copy()
    
    print(f"Validation data: {len(df_valid):,} records")
    print(f"Date range: {df_valid['自然日'].min()} to {df_valid['自然日'].max()}")
    print(f"Tickers: {df_valid['ticker'].unique()}")
    
    # Apply enhanced feature engineering (keep time info for visualization)
    print("🔧 Applying leakage-free feature cleaning...")
    
    # Keep visualization needed time and ID columns
    visualization_cols = ['自然日', 'ticker', '交易所委托号', 'y_label', 
                         '委托_datetime', '事件_datetime', '时间_委托', '时间_事件']
    viz_backup = df_valid[[col for col in visualization_cols if col in df_valid.columns]].copy()
    
    try:
        # 尝试导入防泄露清理函数
        import sys
        from pathlib import Path
        leakage_path = Path(__file__).parent.parent / "data_process" / "features"
        sys.path.append(str(leakage_path))
        
        from leakage_free_features import clean_features_for_training
        df_valid_clean = clean_features_for_training(df_valid, "y_label")
        
        # 将重要的ID和时间列重新添加到清理后的数据中
        df_valid = pd.concat([viz_backup, df_valid_clean], axis=1)
        print("✅ Leakage-free features applied from data process script")
        print(f"   Restored visualization columns: {list(viz_backup.columns)}")
        
    except ImportError as e:
        print(f"⚠️ Could not import leakage cleaning functions: {e}")
        print("   Using basic feature cleaning...")
        
        # 基础清理（保留时间信息）
        leakage_cols = [
            "存活时间_ms", "成交价格", "成交数量", "事件类型",
            "is_cancel_event", "is_trade_event",
            "flag_R1", "flag_R2", "enhanced_spoofing_liberal", 
            "enhanced_spoofing_moderate", "enhanced_spoofing_strict"
        ]
        
        # 移除泄露列（保留重要的可视化列）
        cols_to_remove = [col for col in leakage_cols if col in df_valid.columns and col not in visualization_cols]
        df_valid = df_valid.drop(columns=cols_to_remove)
        print(f"   Removed {len(cols_to_remove)} leakage columns")
        print(f"   Kept visualization columns: {[col for col in visualization_cols if col in df_valid.columns]}")
    
    return df_valid


def load_market_data(data_root, date, ticker):
    """加载指定日期和股票的行情数据"""
    market_file = os.path.join(data_root, "base_data", str(date), str(date), ticker, "行情.csv")
    
    if not os.path.exists(market_file):
        raise FileNotFoundError(f"Market data not found: {market_file}")
    
    # 尝试不同的编码
    encodings = ['gbk', 'utf-8', 'gb2312', 'cp936']
    df_market = None
    
    for encoding in encodings:
        try:
            df_market = pd.read_csv(market_file, encoding=encoding)
            print(f"✅ Successfully loaded market data with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if df_market is None:
        raise ValueError(f"Could not decode market data file with any encoding")
    
    return df_market


def load_order_data(data_root, date, ticker):
    """加载指定日期和股票的逐笔委托数据"""
    order_file = os.path.join(data_root, "base_data", str(date), str(date), ticker, "逐笔委托.csv")
    
    if not os.path.exists(order_file):
        print(f"⚠️ Order data not found: {order_file}")
        return None
    
    # 尝试不同的编码
    encodings = ['gbk', 'utf-8', 'gb2312', 'cp936']
    df_order = None
    
    for encoding in encodings:
        try:
            df_order = pd.read_csv(order_file, encoding=encoding)
            print(f"✅ Successfully loaded order data with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if df_order is None:
        print(f"⚠️ Could not decode order data file")
        return None
    
    return df_order


def prepare_time_series(df_market):
    """准备时间序列数据"""
    # 假设时间列是第4列
    time_col = df_market.columns[3]
    
    # 如果时间是数字格式，转换为时间
    if df_market[time_col].dtype in ['int64', 'float64']:
        time_strs = df_market[time_col].astype(str).str.zfill(8)
        time_strs = time_strs.str[:2] + ':' + time_strs.str[2:4] + ':' + time_strs.str[4:6] + '.' + time_strs.str[6:]
        
        date_col = df_market.columns[2]
        date_str = str(df_market[date_col].iloc[0])
        
        datetime_strs = date_str + ' ' + time_strs
        df_market['datetime'] = pd.to_datetime(datetime_strs, format='%Y%m%d %H:%M:%S.%f', errors='coerce')
    else:
        df_market['datetime'] = pd.to_datetime(df_market[time_col], errors='coerce')
    
    # 移除无效时间
    df_market = df_market.dropna(subset=['datetime'])
    
    # 处理重复时间标签问题
    if df_market['datetime'].duplicated().any():
        print(f"   Warning: Found {df_market['datetime'].duplicated().sum()} duplicate timestamps, removing duplicates")
        df_market = df_market.drop_duplicates(subset=['datetime'], keep='first')
    
    # 重置索引以避免重复索引问题
    df_market = df_market.reset_index(drop=True)
    
    # 按时间排序
    df_market = df_market.sort_values('datetime')
    
    return df_market


def get_price_column(df_market):
    """获取价格列"""
    price_candidates = []
    
    for col in df_market.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['价格', 'price', '成交价', '中间价', '最新价']):
            if df_market[col].dtype in ['int64', 'float64'] and df_market[col].sum() > 0:
                price_candidates.append(col)
    
    if price_candidates:
        for col in price_candidates:
            if '成交' in str(col) or 'trade' in str(col).lower():
                return col
        return price_candidates[0]
    
    # 使用第5列
    if len(df_market.columns) > 4:
        return df_market.columns[4]
    
    raise ValueError("Could not identify price column in market data")


def map_orders_to_time(df_validation, df_order=None, order_id_col='交易所委托号'):
    """Use existing time information in validation set directly, no need to remap"""
    df_result = df_validation.copy()
    
    # Check if time columns exist
    time_columns = ['委托_datetime', '事件_datetime', '时间_委托', '时间_事件']
    available_time_cols = [col for col in time_columns if col in df_result.columns]
    
    print(f"🕒 Detected time columns: {available_time_cols}")
    
    if '委托_datetime' in df_result.columns:
        # Use order time as primary timestamp
        df_result['order_time'] = pd.to_datetime(df_result['委托_datetime'])
        mapped_count = df_result['order_time'].notna().sum()
        print(f"✅ Using 委托_datetime: {mapped_count:,} / {len(df_result):,} orders have timestamps")
        
    elif '事件_datetime' in df_result.columns:
        # Alternative: use event time
        df_result['order_time'] = pd.to_datetime(df_result['事件_datetime'])
        mapped_count = df_result['order_time'].notna().sum()
        print(f"✅ Using 事件_datetime: {mapped_count:,} / {len(df_result):,} orders have timestamps")
        
    elif '时间_委托' in df_result.columns and '自然日' in df_result.columns:
        # Construct timestamp from numeric time
        date_str = df_result['自然日'].astype(str)
        time_str = df_result['时间_委托'].astype(str).str.zfill(9)
        datetime_str = date_str + time_str
        df_result['order_time'] = pd.to_datetime(datetime_str, format='%Y%m%d%H%M%S%f', errors='coerce')
        mapped_count = df_result['order_time'].notna().sum()
        print(f"✅ Constructed from 时间_委托: {mapped_count:,} / {len(df_result):,} orders have timestamps")
        
    else:
        print("⚠️ No available time information found, using virtual timestamps")
        # Create virtual time series
        df_result['order_time'] = pd.date_range(
            start='2025-01-01 09:30:00', 
            periods=len(df_result), 
            freq='1S'
        )
        mapped_count = len(df_result)
    
    return df_result


def make_predictions(model, df_validation, features):
    """使用模型进行预测"""
    print(f"🔮 Making predictions...")
    
    # 检查哪些特征在数据中存在
    available_features = [f for f in features if f in df_validation.columns]
    missing_features = [f for f in features if f not in df_validation.columns]
    
    print(f"   Available features: {len(available_features)}/{len(features)}")
    if missing_features:
        print(f"   Missing features: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
    
    # 创建特征数据，缺失的特征用0填充
    feature_data = pd.DataFrame()
    for feature in features:
        if feature in df_validation.columns:
            feature_data[feature] = df_validation[feature]
        else:
            feature_data[feature] = 0
    
    feature_data = feature_data.fillna(0)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(feature_data)[:, 1]
    else:
        y_pred_scores = model.predict(feature_data)
        y_pred_proba = 1 / (1 + np.exp(-y_pred_scores))
    
    df_result = df_validation.copy()
    df_result['predicted_proba'] = y_pred_proba
    df_result['predicted_binary'] = (y_pred_proba > 0.5).astype(int)
    
    # 计算统计值（避免Series转换问题）
    try:
        true_positive = int(df_result['y_label'].sum())
    except (ValueError, TypeError):
        true_positive = int(df_result['y_label'].values.sum())
    
    try:
        pred_positive = int(df_result['predicted_binary'].sum())
    except (ValueError, TypeError):
        pred_positive = int(df_result['predicted_binary'].values.sum())
    
    print(f"✅ Predictions completed")
    print(f"   True positives: {true_positive:,}")
    print(f"   Predicted positives: {pred_positive:,}")
    print(f"   Max prediction probability: {float(y_pred_proba.max()):.4f}")
    print(f"   Mean prediction probability: {float(y_pred_proba.mean()):.4f}")
    
    return df_result


def enhance_features_simple(df):
    """简化的特征增强（仅生成基础增强特征）"""
    print("🔧 Generating basic enhanced features...")
    
    # 计算基础衍生特征
    if 'relative_spread' not in df.columns and 'spread' in df.columns and 'mid_price' in df.columns:
        df['relative_spread'] = df['spread'] / df['mid_price']
    
    # 技术指标特征
    if 'mid_price' in df.columns:
        df['price_volatility'] = df.groupby('ticker')['mid_price'].transform(
            lambda x: x.rolling(100, min_periods=1).std()
        )
        df['price_momentum'] = df.groupby('ticker')['mid_price'].transform(
            lambda x: x.pct_change(5)
        )
    
    if 'spread' in df.columns:
        df['spread_volatility'] = df.groupby('ticker')['spread'].transform(
            lambda x: x.rolling(50, min_periods=1).std()
        )
    
    # 订单流指标
    if 'bid1' in df.columns and 'ask1' in df.columns:
        df['order_imbalance'] = (df['bid1'] - df['ask1']) / (df['bid1'] + df['ask1'])
    else:
        df['order_imbalance'] = 0
    
    # 历史行为特征
    if 'log_qty' in df.columns:
        df['hist_order_size_mean'] = df.groupby('ticker')['log_qty'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
    
    if 'spread' in df.columns:
        df['hist_spread_mean'] = df.groupby('ticker')['spread'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
    
    # 价格级别特征
    if 'delta_mid' in df.columns and 'spread' in df.columns:
        df['at_bid'] = (df['delta_mid'] <= -df['spread']/2).astype(int)
        df['at_ask'] = (df['delta_mid'] >= df['spread']/2).astype(int)
        df['between_quotes'] = ((df['delta_mid'] > -df['spread']/2) & 
                               (df['delta_mid'] < df['spread']/2)).astype(int)
    else:
        df['at_bid'] = 0
        df['at_ask'] = 0
        df['between_quotes'] = 0
    
    # 分位数特征
    for col in ['log_qty', 'spread', 'delta_mid']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('ticker')[col].transform(
                lambda x: x.rank(pct=True)
            )
            df[f'{col}_zscore'] = df.groupby('ticker')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    
    # 处理无穷值和缺失值
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    print(f"✅ Enhanced features created, final shape: {df.shape}")
    return df


def plot_market_with_anomalies(df_market, df_predictions, ticker, date, output_dir, 
                              prob_threshold=0.1, top_k_percent=0.05):
    """Plot market data with anomaly indicators"""
    print(f"📈 Creating market visualization...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df_market = prepare_time_series(df_market)
        price_col = get_price_column(df_market)
        
        # 确保数据不为空且有有效时间序列
        if len(df_market) == 0 or df_market['datetime'].isna().all():
            raise ValueError("No valid time series data available")
        
        # 确保索引和列名唯一
        if df_market.index.duplicated().any():
            df_market = df_market.reset_index(drop=True)
        
        if df_market.columns.duplicated().any():
            df_market = df_market.loc[:, ~df_market.columns.duplicated()]
            
        # 同样处理预测数据
        df_predictions = df_predictions.reset_index(drop=True)
        if df_predictions.columns.duplicated().any():
            df_predictions = df_predictions.loc[:, ~df_predictions.columns.duplicated()]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
        
        # Main plot: price trend
        ax1.plot(df_market['datetime'], df_market[price_col], 
                 color='black', linewidth=1, alpha=0.8, label='Price Trend')
        
        # Mark real anomalies
        true_anomalies = df_predictions[df_predictions['y_label'] == 1]
        if len(true_anomalies) > 0:
            if 'order_time' in true_anomalies.columns and true_anomalies['order_time'].notna().any():
                true_times = pd.to_datetime(true_anomalies['order_time'].dropna())
            else:
                market_start = df_market['datetime'].min()
                market_end = df_market['datetime'].max()
                time_range = (market_end - market_start).total_seconds()
                n_orders = len(true_anomalies)
                estimated_times = [market_start + timedelta(seconds=i*time_range/n_orders) 
                                  for i in range(n_orders)]
                true_times = pd.Series(estimated_times)
            
            # Use flag to ensure label appears only once
            real_anomaly_labeled = False
            for time_point in true_times:
                ax1.axvline(x=time_point, color='red', alpha=0.7, linewidth=2, 
                           label='Real Anomaly' if not real_anomaly_labeled else "")
                real_anomaly_labeled = True
        
        # Mark predicted anomalies
        top_k_threshold = df_predictions['predicted_proba'].quantile(1 - top_k_percent)
        pred_anomalies = df_predictions[df_predictions['predicted_proba'] >= top_k_threshold]
        
        if len(pred_anomalies) > 0:
            if 'order_time' in pred_anomalies.columns and pred_anomalies['order_time'].notna().any():
                pred_times = pd.to_datetime(pred_anomalies['order_time'].dropna())
            else:
                market_start = df_market['datetime'].min()
                market_end = df_market['datetime'].max()
                time_range = (market_end - market_start).total_seconds()
                n_orders = len(pred_anomalies)
                estimated_times = [market_start + timedelta(seconds=i*time_range/n_orders) 
                                  for i in range(n_orders)]
                pred_times = pd.Series(estimated_times)
            
            # Use flag to ensure label appears only once
            pred_anomaly_labeled = False
            for time_point in pred_times:
                ax1.axvline(x=time_point, color='blue', alpha=0.5, linewidth=1.5, linestyle='--',
                           label='Predicted Anomaly' if not pred_anomaly_labeled else "")
                pred_anomaly_labeled = True
        
        ax1.set_title(f'{ticker} Market Trend and Anomaly Detection - {date}', fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'Price ({price_col})', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 子图：预测概率分布
        if len(df_predictions) > 0:
            if 'order_time' in df_predictions.columns and df_predictions['order_time'].notna().any():
                df_pred_sorted = df_predictions.dropna(subset=['order_time']).sort_values('order_time')
                pred_times_all = pd.to_datetime(df_pred_sorted['order_time'])
                pred_probas = df_pred_sorted['predicted_proba']
            else:
                market_start = df_market['datetime'].min()
                market_end = df_market['datetime'].max()
                time_range = (market_end - market_start).total_seconds()
                n_orders = len(df_predictions)
                estimated_times = [market_start + timedelta(seconds=i*time_range/n_orders) 
                                  for i in range(n_orders)]
                pred_times_all = pd.Series(estimated_times)
                pred_probas = df_predictions['predicted_proba']
            
            ax2.plot(pred_times_all, pred_probas, color='green', alpha=0.6, linewidth=1)
            ax2.fill_between(pred_times_all, pred_probas, alpha=0.3, color='green')
            
            ax2.axhline(y=prob_threshold, color='orange', linestyle=':', alpha=0.7, 
                       label=f'Probability Threshold {prob_threshold}')
            ax2.axhline(y=top_k_threshold, color='purple', linestyle=':', alpha=0.7,
                       label=f'Top {top_k_percent*100:.1f}% Threshold {top_k_threshold:.3f}')
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Prediction Probability', fontsize=12)
        ax2.set_title('Anomaly Probability Prediction', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylim(0, max(1, df_predictions['predicted_proba'].max() * 1.1))
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{ticker}_{date}_market_anomaly_detection.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Market visualization saved: {output_file}")
        
        plt.close()  # 关闭图形以释放内存
        
        return fig
        
    except Exception as e:
        plt.close('all')  # 清理任何打开的图形
        raise RuntimeError(f"Failed to create market visualization: {str(e)}")


def generate_summary_report(df_predictions, model_data, output_dir):
    """Generate prediction results summary report"""
    print("📋 Generating summary report...")
    
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    y_true = df_predictions['y_label']
    y_pred_proba = df_predictions['predicted_proba']
    y_pred_binary = df_predictions['predicted_binary']
    
    # 确保输入是正确的数组格式
    y_true_array = np.asarray(y_true).ravel()  # 展平为1D数组
    y_pred_proba_array = np.asarray(y_pred_proba).ravel()  # 展平为1D数组
    
    # 确保数组长度一致
    min_len = min(len(y_true_array), len(y_pred_proba_array))
    y_true_array = y_true_array[:min_len]
    y_pred_proba_array = y_pred_proba_array[:min_len]
    
    print(f"   Array lengths after alignment: y_true={len(y_true_array)}, y_pred={len(y_pred_proba_array)}")
    
    # 确保数组形状正确
    if y_true_array.ndim != 1 or y_pred_proba_array.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    
    pr_auc = average_precision_score(y_true_array, y_pred_proba_array)
    roc_auc = roc_auc_score(y_true_array, y_pred_proba_array)
    
    total_samples = len(df_predictions)
    
    # 安全地转换Series为int（避免转换错误）
    try:
        true_positives = int(y_true_array.sum())
    except (ValueError, TypeError):
        true_positives = int(np.asarray(y_true).sum())
        
    try:
        pred_positives = int(y_pred_binary.sum())
    except (ValueError, TypeError):
        pred_positives = int(np.asarray(y_pred_binary).sum())
    
    precision_at_k = {}
    for k in [0.001, 0.005, 0.01, 0.05, 0.1]:
        k_samples = max(1, int(total_samples * k))
        top_k_idx = y_pred_proba.nlargest(k_samples).index
        prec_k = np.asarray(y_true.loc[top_k_idx]).mean()
        precision_at_k[f"P@{k*100:.1f}%"] = prec_k
    
    report = f"""# Model Prediction Summary Report

## Model Information
- **Model Type**: {model_data.get('model_type', 'Unknown')}
- **Feature Count**: {len(model_data['features'])}
- **Training Performance**: {model_data.get('results', {}).get('PR-AUC', 'N/A')}

## Validation Set Prediction Results

### Basic Statistics
- **Total Samples**: {total_samples:,}
- **Real Anomalies**: {true_positives:,} ({true_positives/total_samples*100:.3f}%)
- **Predicted Anomalies**: {pred_positives:,} ({pred_positives/total_samples*100:.3f}%)

### Performance Metrics
- **PR-AUC**: {pr_auc:.6f}
- **ROC-AUC**: {roc_auc:.6f}

### Precision@K Analysis
"""
    
    for metric, value in precision_at_k.items():
        report += f"- **{metric}**: {value:.6f}\n"
    
    report += f"""
### Prediction Probability Distribution
- **Minimum**: {y_pred_proba.min():.6f}
- **Maximum**: {y_pred_proba.max():.6f}
- **Mean**: {y_pred_proba.mean():.6f}
- **Median**: {y_pred_proba.median():.6f}
- **Standard Deviation**: {y_pred_proba.std():.6f}
"""
    
    if 'ticker' in df_predictions.columns:
        try:
            ticker_stats = df_predictions.groupby('ticker').agg({
                'y_label': ['count', 'sum'],
                'predicted_proba': ['mean', 'max'],
                'predicted_binary': 'sum'
            }).round(4)
            
            report += "\n### Results by Stock\n\n" + ticker_stats.to_string() + "\n"
        except Exception as e:
            print(f"⚠️ Stock grouping statistics failed: {e}")
            # Simplified stock statistics
            ticker_simple_stats = df_predictions.groupby('ticker').size()
            report += f"\n### Sample Count by Stock\n\n{ticker_simple_stats.to_string()}\n"
    
    report_file = os.path.join(output_dir, "prediction_summary_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Summary report saved: {report_file}")
    
    return report


def create_simple_summary_plot(df_predictions, ticker, date, output_dir):
    """Create simple statistical summary plots"""
    print(f"📊 Creating simple summary visualization...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 确保数据没有重复索引和重复列
        df_predictions = df_predictions.reset_index(drop=True)
        
        # 检查并处理重复列名
        if df_predictions.columns.duplicated().any():
            df_predictions = df_predictions.loc[:, ~df_predictions.columns.duplicated()]
            print(f"   Removed duplicate columns")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Prediction probability distribution histogram
        ax1.hist(df_predictions['predicted_proba'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Prediction Probability Distribution', fontsize=12)
        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot of real labels vs prediction probabilities
        true_anomalies = df_predictions[df_predictions['y_label'] == 1]['predicted_proba']
        normal_samples = df_predictions[df_predictions['y_label'] == 0]['predicted_proba']
        
        ax2.boxplot([normal_samples, true_anomalies], labels=['Normal', 'Anomaly'])
        ax2.set_title('Prediction Probability Distribution by Real Label', fontsize=12)
        ax2.set_ylabel('Prediction Probability')
        ax2.grid(True, alpha=0.3)
        
        # 3. Top K 精度分析
        k_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        precisions = []
        
        for k in k_values:
            k_samples = max(1, int(len(df_predictions) * k))
            top_k_idx = df_predictions['predicted_proba'].nlargest(k_samples).index
            prec_k = df_predictions.loc[top_k_idx, 'y_label'].mean()
            precisions.append(prec_k)
        
        ax3.plot([k*100 for k in k_values], precisions, 'bo-', linewidth=2, markersize=8)
        ax3.set_title('Precision@K Analysis', fontsize=12)
        ax3.set_xlabel('Top K%')
        ax3.set_ylabel('Precision')
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistical summary
        stats_text = f"""Sample Statistics:
Total Samples: {len(df_predictions):,}
Real Anomalies: {int(df_predictions['y_label'].sum()):,}
Anomaly Rate: {df_predictions['y_label'].mean()*100:.3f}%

Prediction Statistics:
Max Probability: {df_predictions['predicted_proba'].max():.4f}
Mean Probability: {df_predictions['predicted_proba'].mean():.4f}
Median Probability: {df_predictions['predicted_proba'].median():.4f}"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax4.set_title('Statistical Summary', fontsize=12)
        ax4.axis('off')
        
        plt.suptitle(f'{ticker} - {date} Prediction Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{ticker}_{date}_simple_summary.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Simple summary saved: {output_file}")
        
        plt.close()
        return fig
        
    except Exception as e:
        plt.close('all')
        print(f"⚠️ Simple visualization failed: {e}")
        return None


def plot_monthly_summary(df_predictions, month_str, output_dir, prob_threshold=0.1, top_k_percent=0.05):
    """生成整个月的汇总图表"""
    print(f"📊 Creating monthly summary visualization for {month_str}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 按日期聚合数据
        daily_stats = df_predictions.groupby('自然日').agg({
            'y_label': ['count', 'sum'],
            'predicted_proba': ['mean', 'max', 'std'],
            'predicted_binary': 'sum'
        }).round(4)
        
        # 展平列名
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
        daily_stats = daily_stats.reset_index()
        
        # 转换日期
        daily_stats['date'] = pd.to_datetime(daily_stats['自然日'].astype(str), format='%Y%m%d')
        daily_stats = daily_stats.sort_values('date')
        
        # 计算每日统计
        daily_stats['anomaly_rate'] = daily_stats['y_label_sum'] / daily_stats['y_label_count']
        daily_stats['pred_anomaly_rate'] = daily_stats['predicted_binary_sum'] / daily_stats['y_label_count']
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 每日真实异常数量和预测异常数量
        ax1.plot(daily_stats['date'], daily_stats['y_label_sum'], 'ro-', 
                label='Real Anomalies', linewidth=2, markersize=6)
        ax1.plot(daily_stats['date'], daily_stats['predicted_binary_sum'], 'bo-', 
                label='Predicted Anomalies', linewidth=2, markersize=6)
        ax1.set_title(f'Daily Anomaly Counts - {month_str}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Anomaly Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 每日异常率
        ax2.plot(daily_stats['date'], daily_stats['anomaly_rate'] * 100, 'r-', 
                label='Real Anomaly Rate', linewidth=2)
        ax2.plot(daily_stats['date'], daily_stats['pred_anomaly_rate'] * 100, 'b--', 
                label='Predicted Anomaly Rate', linewidth=2)
        ax2.set_title(f'Daily Anomaly Rates - {month_str}', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Anomaly Rate (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 每日平均预测概率
        ax3.plot(daily_stats['date'], daily_stats['predicted_proba_mean'], 'g-', 
                linewidth=2, label='Mean Prediction Probability')
        ax3.fill_between(daily_stats['date'], 
                        daily_stats['predicted_proba_mean'] - daily_stats['predicted_proba_std'],
                        daily_stats['predicted_proba_mean'] + daily_stats['predicted_proba_std'],
                        alpha=0.3, color='green')
        ax3.axhline(y=prob_threshold, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Threshold {prob_threshold}')
        ax3.set_title(f'Daily Average Prediction Probability - {month_str}', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Prediction Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. 月度统计总结
        total_samples = len(df_predictions)
        total_real_anomalies = int(df_predictions['y_label'].sum())
        total_pred_anomalies = int(df_predictions['predicted_binary'].sum())
        overall_anomaly_rate = total_real_anomalies / total_samples * 100
        
        # 按股票统计
        if 'ticker' in df_predictions.columns:
            ticker_stats = df_predictions.groupby('ticker').agg({
                'y_label': ['count', 'sum'],
                'predicted_proba': 'mean'
            }).round(4)
            ticker_stats.columns = ['_'.join(col).strip() for col in ticker_stats.columns.values]
            
            # 柱状图显示各股票的异常情况
            tickers = ticker_stats.index.tolist()
            real_counts = ticker_stats['y_label_sum'].tolist()
            
            x_pos = np.arange(len(tickers))
            ax4.bar(x_pos, real_counts, alpha=0.7, color='red', label='Real Anomalies')
            ax4.set_xlabel('Stock Ticker')
            ax4.set_ylabel('Anomaly Count')
            ax4.set_title(f'Anomaly Count by Stock - {month_str}', fontsize=14, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(tickers, rotation=45)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            # 如果没有股票信息，显示文本统计
            stats_text = f"""Monthly Summary Statistics:

Total Samples: {total_samples:,}
Real Anomalies: {total_real_anomalies:,}
Predicted Anomalies: {total_pred_anomalies:,}
Overall Anomaly Rate: {overall_anomaly_rate:.3f}%

Trading Days: {len(daily_stats)}
Avg Daily Samples: {total_samples/len(daily_stats):.0f}
Avg Daily Anomalies: {total_real_anomalies/len(daily_stats):.1f}"""
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
            ax4.set_title(f'Monthly Statistics Summary - {month_str}', fontsize=14, fontweight='bold')
            ax4.axis('off')
        
        plt.suptitle(f'Monthly Anomaly Detection Summary - {month_str}', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'monthly_summary_{month_str}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Monthly summary saved: {output_file}")
        
        plt.close()
        
        # 生成月度汇总报告
        monthly_report = f"""# Monthly Summary Report - {month_str}

## Overall Statistics
- **Total Samples**: {total_samples:,}
- **Real Anomalies**: {total_real_anomalies:,} ({overall_anomaly_rate:.3f}%)
- **Predicted Anomalies**: {total_pred_anomalies:,}
- **Trading Days**: {len(daily_stats)}

## Daily Statistics
- **Average Daily Samples**: {total_samples/len(daily_stats):.0f}
- **Average Daily Anomalies**: {total_real_anomalies/len(daily_stats):.1f}
- **Max Daily Anomalies**: {daily_stats['y_label_sum'].max()}
- **Min Daily Anomalies**: {daily_stats['y_label_sum'].min()}

## Performance Summary
- **Average Prediction Probability**: {df_predictions['predicted_proba'].mean():.6f}
- **Max Prediction Probability**: {df_predictions['predicted_proba'].max():.6f}
- **Prediction Standard Deviation**: {df_predictions['predicted_proba'].std():.6f}
"""
        
        report_file = os.path.join(output_dir, f'monthly_summary_report_{month_str}.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(monthly_report)
        
        print(f"✅ Monthly summary report saved: {report_file}")
        
        return fig
        
    except Exception as e:
        plt.close('all')
        raise RuntimeError(f"Failed to create monthly summary: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Model Prediction and Visualization Analysis")
    parser.add_argument("--data_root", required=True, help="Data root directory")
    parser.add_argument("--model_path", required=True, help="Path to trained model file")
    parser.add_argument("--valid_regex", default="202505", help="Validation set date regex")
    parser.add_argument("--output_dir", default="results/prediction_analysis", help="Output directory")
    parser.add_argument("--prob_threshold", type=float, default=0.1, help="Anomaly probability threshold")
    parser.add_argument("--top_k_percent", type=float, default=0.05, help="Top K% prediction display ratio")
    parser.add_argument("--max_plots", type=int, default=5, help="Maximum number of plots")
    
    args = parser.parse_args()
    
    print("🚀 Starting model prediction and visualization analysis")
    print(f"   Data root: {args.data_root}")
    print(f"   Model: {args.model_path}")
    print(f"   Validation regex: {args.valid_regex}")
    print(f"   Output: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 1. Load model
        model, features, model_data = load_model(args.model_path)
        
        # 2. Load validation data
        df_validation = load_validation_data(args.data_root, args.valid_regex)
        
        # 3. Make predictions
        df_predictions = make_predictions(model, df_validation, features)
        
        # 4. Group by date and stock for visualization
        ticker_date_groups = df_predictions.groupby(['ticker', '自然日'])
        plot_count = 0
        
        for (ticker, date), group_data in ticker_date_groups:
            if plot_count >= args.max_plots:
                print(f"📊 Reached maximum plot limit ({args.max_plots})")
                break
            
            print(f"\n📈 Processing: {ticker} - {date}")
            print(f"   Sample count: {len(group_data):,}")
            
            # Calculate statistics (avoid Series conversion issues)
            try:
                true_positive = int(group_data['y_label'].sum())
            except (ValueError, TypeError):
                true_positive = int(group_data['y_label'].values.sum())
            
            try:
                high_prob_pred = int((group_data['predicted_proba'] > args.prob_threshold).sum())
            except (ValueError, TypeError):
                high_prob_pred = int((group_data['predicted_proba'] > args.prob_threshold).values.sum())
            
            print(f"   Real anomalies: {true_positive:,}")
            print(f"   High probability predictions: {high_prob_pred:,}")
            
            try:
                df_market = load_market_data(args.data_root, date, ticker)
                df_order = load_order_data(args.data_root, date, ticker)
                
                group_data_with_time = map_orders_to_time(group_data, df_order)
                
                try:
                    plot_market_with_anomalies(
                        df_market, group_data_with_time, ticker, date, args.output_dir,
                        prob_threshold=args.prob_threshold, top_k_percent=args.top_k_percent
                    )
                except Exception as market_plot_error:
                    print(f"⚠️ Market plot failed, creating simple summary instead: {market_plot_error}")
                    create_simple_summary_plot(group_data, ticker, date, args.output_dir)
                
                plot_count += 1
                
            except Exception as e:
                print(f"⚠️ Skipping {ticker}-{date}: {e}")
                continue
        
        # 5. Generate summary report
        generate_summary_report(df_predictions, model_data, args.output_dir)
        
        # 6. 生成月度汇总图表
        print(f"\n📊 Generating monthly summary...")
        month_str = args.valid_regex
        try:
            plot_monthly_summary(df_predictions, month_str, args.output_dir,
                                prob_threshold=args.prob_threshold, top_k_percent=args.top_k_percent)
        except Exception as e:
            print(f"⚠️ Monthly summary failed: {e}")
        
        print(f"\n✅ Analysis complete!")
        print(f"   Generated {plot_count} daily visualization plots")
        print(f"   Generated monthly summary")
        print(f"   Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Usage example:
--prob_threshold (默认: 0.1)
作用: 异常概率阈值
说明: 超过此概率的预测被标记为异常，用于可视化中的阈值线
取值范围: 0.0 - 1.0
示例: 0.1 (10%的概率阈值)
--top_k_percent (默认: 0.05)
作用: Top K%预测显示比例
说明: 显示概率最高的前K%预测作为异常标记
取值范围: 0.0 - 1.0
示例: 0.05 (显示前5%的高概率预测)
--max_plots (默认: 5)
作用: 最大生成图表数量
说明: 限制生成的股票-日期组合图表数量，避免生成过多文件
示例: 5 (最多生成5个图表)

# 生成整个月的分析（每日单独图表 + 月度汇总图表）
python scripts/analysis/model_prediction_visualization.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --model_path "results/trained_models/spoofing_model_Enhanced_undersample_Ensemble.pkl" \
  --valid_regex "202505" \
  --output_dir "results/prediction_visualization/202505_full_month" \
  --prob_threshold 0.01 \
  --top_k_percent 0.005 \
  --max_plots 50

输出文件说明:
1. 每个股票-日期组合的单独图表: ticker_date_market_anomaly_detection.png
2. 每个股票-日期的简单汇总: ticker_date_simple_summary.png  
3. 月度汇总图表: monthly_summary_YYYYMM.png (包含4个子图)
   - 每日异常数量趋势
   - 每日异常率变化
   - 每日平均预测概率
   - 按股票的异常统计
4. 月度汇总报告: monthly_summary_report_YYYYMM.md
5. 整体预测报告: prediction_summary_report.md
""" 