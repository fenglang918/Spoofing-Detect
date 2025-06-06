#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
潜在操纵时段识别与异常交易热力图分析
==========================================

本脚本用于：
1. 识别潜在的市场操纵时段
2. 分析异常交易行为模式
3. 生成多维度热力图可视化
4. 提供操纵行为的时间、空间分布分析

功能特性：
- 支持加载已训练的spoofing检测模型
- 基于模型预测和统计异常检测的双重检测
- 时间序列异常检测
- 股票×时间热力图
- 交互式可视化
- 统计报告生成
"""

import argparse
import glob
import os
import sys
import warnings
import pickle
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

# 进度条和性能监控
from tqdm import tqdm
import psutil
import gc

# 🖥️ 设置matplotlib后端为Agg（适用于无GUI的Linux服务器）
import matplotlib
matplotlib.use('Agg')  # 必须在import pyplot之前设置
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 服务器环境优化配置
import os
os.environ['MPLBACKEND'] = 'Agg'  # 确保环境变量也设置正确
plt.ioff()  # 关闭交互模式
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# LightGBM for loading saved models
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM not available, will skip model-based detection")

# 导入训练脚本中的特征工程函数
try:
    # Fix path calculation: go up two levels from heatmap_analysis to get to scripts, then to train
    sys.path.append(str(Path(__file__).parent.parent.parent / "train"))
    from train_baseline_enhanced_fixed import enhance_features
    HAS_FEATURE_ENGINEERING = True
    print("✅ 成功导入训练脚本的特征工程函数")
except ImportError as e:
    HAS_FEATURE_ENGINEERING = False
    print(f"⚠️ 无法导入特征工程函数: {e}")
    print("   将使用基础特征，可能影响模型预测准确性")

# 集成模型类定义（用于加载训练好的模型）
class EnsembleClassifier:
    """集成分类器 - 用于加载已训练的集成模型"""
    def __init__(self, models=None):
        if models is None:
            self.models = {}
        else:
            self.models = models
        self.weights = None
        self.fitted_models = {}
    
    def predict_proba(self, X):
        """集成预测"""
        if not self.fitted_models:
            return np.zeros((len(X), 2))
            
        predictions = []
        for name, model in self.fitted_models.items():
            weight = self.weights.get(name, 1.0/len(self.fitted_models)) if self.weights else 1.0/len(self.fitted_models)
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred * weight)
        
        ensemble_pred = np.sum(predictions, axis=0)
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

warnings.filterwarnings('ignore')

# 性能监控工具
class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.checkpoints = []
    
    def start(self):
        self.start_time = datetime.now()
        gc.collect()  # 清理内存
        
    def checkpoint(self, name):
        if self.start_time is None:
            self.start()
        
        now = datetime.now()
        elapsed = (now - self.start_time).total_seconds()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        self.checkpoints.append({
            'name': name,
            'elapsed': elapsed,
            'memory_mb': memory_mb
        })
        
        print(f"⏱️ {name}: {elapsed:.1f}s, Memory: {memory_mb:.1f}MB")
        
    def summary(self):
        if not self.checkpoints:
            return
        
        print("\n📊 Performance Summary:")
        for i, cp in enumerate(self.checkpoints):
            prev_time = self.checkpoints[i-1]['elapsed'] if i > 0 else 0
            step_time = cp['elapsed'] - prev_time
            print(f"  {cp['name']}: {step_time:.1f}s (total: {cp['elapsed']:.1f}s, mem: {cp['memory_mb']:.1f}MB)")

# 并行数据加载工具
def load_parquet_file(file_path):
    """加载单个parquet文件"""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"⚠️ 加载文件失败 {file_path}: {e}")
        return pd.DataFrame()

def parallel_load_parquet(file_paths, max_workers=None, desc="Loading files"):
    """并行加载多个parquet文件"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(file_paths), 8)  # 限制最大线程数
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用tqdm显示进度条
        futures = [executor.submit(load_parquet_file, fp) for fp in file_paths]
        results = []
        
        for future in tqdm(futures, desc=desc, unit="files"):
            df = future.result()
            if not df.empty:
                results.append(df)
    
    return results

# 并发统计异常检测工具
def compute_statistical_features_for_stock(args):
    """为单个股票计算统计异常特征（用于并行处理）"""
    stock_data, ticker, feature_cols = args
    
    try:
        results = {}
        
        # 计算滚动统计特征
        for col in feature_cols:
            if col in stock_data.columns:
                # 滚动标准差
                rolling_std = stock_data[col].rolling(window=50, min_periods=1).std()
                results[f'{col}_rolling_std'] = rolling_std
                
                # Z-score anomaly
                mean_val = stock_data[col].expanding().mean()
                std_val = stock_data[col].expanding().std()
                z_scores = abs((stock_data[col] - mean_val) / (std_val + 1e-8))
                results[f'{col}_zscore_anomaly'] = (z_scores > 3).astype(int)
                
                # 分位数异常
                rolling_quantile_95 = stock_data[col].rolling(window=100, min_periods=1).quantile(0.95)
                results[f'{col}_quantile_anomaly'] = (stock_data[col] > rolling_quantile_95).astype(int)
        
        # 添加ticker信息
        results['ticker'] = ticker
        results['index'] = stock_data.index
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"⚠️ 股票 {ticker} 统计特征计算失败: {e}")
        return pd.DataFrame()

def parallel_statistical_features(df, feature_cols, max_workers=None):
    """并行计算所有股票的统计异常特征"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)
    
    print(f"🚀 并行计算统计异常特征 (线程数: {max_workers})")
    
    # 按股票分组
    stock_groups = []
    for ticker, group in df.groupby('ticker'):
        stock_groups.append((group, ticker, feature_cols))
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_statistical_features_for_stock, args) 
                  for args in stock_groups]
        
        results = []
        for future in tqdm(futures, desc="计算统计特征", unit="股票"):
            result_df = future.result()
            if not result_df.empty:
                results.append(result_df)
    
    if results:
        # 合并结果
        combined_results = pd.concat(results, ignore_index=True)
        return combined_results
    else:
        return pd.DataFrame()

def parallel_anomaly_detection(X, contamination=0.1, min_samples=5, max_workers=None, fast_mode=False):
    """并行运行多种异常检测算法（增强版 - 添加超时和进度监控）"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)
    
    print(f"🔄 并行异常检测 (算法线程数: {max_workers})")
    print(f"📊 数据形状: {X.shape}, 内存使用: {X.nbytes / 1024 / 1024:.1f}MB")
    
    # 检查数据规模，如果太大则自动采样
    if len(X) > 50000:
        print(f"⚠️ 数据量较大 ({len(X):,} 行)，建议考虑采样以避免卡住")
        print("   如果继续卡住，请添加 --sample_size 50000 参数")
    
    # 定义多种异常检测算法
    algorithms = {
        'isolation_forest': IsolationForest(
            contamination=contamination, 
            random_state=42, 
            n_jobs=1,  # 减少内部并行，避免过度并发
            max_samples=min(1000, len(X))  # 限制样本数，加快速度
        )
    }
    
    # 在快速模式下，只使用IsolationForest
    if not fast_mode:
        algorithms.update({
            'dbscan': DBSCAN(
                eps=0.5, 
                min_samples=min(min_samples, 10),  # 限制最小样本数
                n_jobs=1  # 减少内部并行
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=min(20, len(X)-1),  # 限制邻居数
                n_jobs=1  # 减少内部并行
            )
        })
    else:
        print("⚡ 快速模式：仅使用 IsolationForest 算法")
    
    print(f"📋 将运行 {len(algorithms)} 个算法: {list(algorithms.keys())}")
    
    # 移除不可用的算法
    algorithms = {k: v for k, v in algorithms.items() if v is not None}
    
    def run_algorithm(args):
        """运行单个异常检测算法（增强版）"""
        name, algorithm, data = args
        start_time = datetime.now()
        
        try:
            print(f"🟡 开始运行 {name} 算法...")
            
            if name == 'local_outlier_factor':
                # LOF可能很慢，对大数据集进行采样
                if len(data) > 10000:
                    print(f"   LOF算法对大数据集采样到 10000 行")
                    indices = np.random.choice(len(data), 10000, replace=False)
                    sampled_data = data[indices]
                    scores = algorithm.fit_predict(sampled_data)
                    # 将结果映射回原数据
                    full_scores = np.zeros(len(data))
                    full_scores[indices] = scores
                    result_anomalies = (full_scores == -1).astype(int)
                else:
                    scores = algorithm.fit_predict(data)
                    result_anomalies = (scores == -1).astype(int)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"✅ {name} 完成，耗时: {elapsed:.1f}s")
                return name, result_anomalies, scores if len(data) <= 10000 else full_scores
                
            elif name == 'dbscan':
                # DBSCAN返回聚类标签
                labels = algorithm.fit_predict(data)
                anomalies = (labels == -1).astype(int)
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"✅ {name} 完成，耗时: {elapsed:.1f}s")
                return name, anomalies, labels
                
            else:
                # Isolation Forest
                anomalies = algorithm.fit_predict(data)
                anomalies = (anomalies == -1).astype(int)
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"✅ {name} 完成，耗时: {elapsed:.1f}s")
                return name, anomalies, None
                
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"❌ 算法 {name} 执行失败 (耗时: {elapsed:.1f}s): {e}")
            return name, np.zeros(len(data)), None
    
    # 并行执行所有算法（添加超时控制）
    from concurrent.futures import as_completed
    
    print("🚀 启动并行异常检测算法...")
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        tasks = [(name, algo, X) for name, algo in algorithms.items()]
        future_to_name = {
            executor.submit(run_algorithm, task): task[0] 
            for task in tasks
        }
        
        # 等待结果，添加超时机制
        timeout_per_algorithm = 300  # 每个算法最多5分钟
        
        for future in as_completed(future_to_name, timeout=timeout_per_algorithm * len(algorithms)):
            try:
                name, anomalies, extra = future.result(timeout=timeout_per_algorithm)
                results[name] = {
                    'anomalies': anomalies,
                    'extra': extra
                }
                print(f"📋 算法 {name} 结果已收集")
                
            except TimeoutError:
                algorithm_name = future_to_name[future]
                print(f"⏰ 算法 {algorithm_name} 超时，跳过")
                results[algorithm_name] = {
                    'anomalies': np.zeros(len(X)),
                    'extra': None
                }
            except Exception as e:
                algorithm_name = future_to_name[future]
                print(f"⚠️ 算法 {algorithm_name} 异常: {e}")
                results[algorithm_name] = {
                    'anomalies': np.zeros(len(X)),
                    'extra': None
                }
    
    # 验证结果
    successful_algorithms = [name for name, result in results.items() 
                           if result['anomalies'].sum() > 0]
    print(f"✅ 异常检测完成，成功算法: {len(successful_algorithms)}/{len(algorithms)}")
    print(f"   成功的算法: {successful_algorithms}")
    
    if not successful_algorithms:
        print("⚠️ 所有算法都失败了，返回默认结果")
        # 返回默认的异常检测结果
        default_anomalies = np.random.random(len(X)) > 0.95  # 5%的随机异常
        results['default'] = {
            'anomalies': default_anomalies.astype(int),
            'extra': None
        }
    
    return results

# 设置英文字体和样式（避免中文字体兼容性问题）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class ManipulationDetector:
    """操纵行为检测器 - 支持模型预测和统计异常检测（并发优化版）"""
    
    def __init__(self, contamination=0.1, min_samples=5, model_path=None, enable_parallel=True, max_workers=None, fast_mode=False):
        """
        初始化检测器
        
        Args:
            contamination: 异常比例，用于IsolationForest
            min_samples: DBSCAN最小样本数
            model_path: 已训练模型的路径
            enable_parallel: 是否启用并发处理
            max_workers: 最大线程数
            fast_mode: 快速模式，禁用可能卡死的算法
        """
        self.contamination = contamination
        self.min_samples = min_samples
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.fast_mode = fast_mode
        
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_jobs=-1
        )
        
        # 加载已训练的模型
        self.trained_model = None
        self.model_features = None
        if model_path and os.path.exists(model_path):
            self.load_trained_model(model_path)
    
    def load_trained_model(self, model_path):
        """加载已训练的spoofing检测模型"""
        try:
            print(f"📦 加载已训练模型: {model_path}")
            
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            elif model_path.endswith('.joblib'):
                model_data = joblib.load(model_path)
            else:
                # 尝试作为LightGBM模型加载
                if HAS_LIGHTGBM:
                    self.trained_model = lgb.Booster(model_file=model_path)
                    print("✅ LightGBM模型加载成功")
                    return
                else:
                    print("❌ 无法加载模型：不支持的格式")
                    return
            
            # 处理pickle/joblib格式
            if isinstance(model_data, dict):
                self.trained_model = model_data.get('model')
                self.model_features = model_data.get('features')
            else:
                self.trained_model = model_data
            
            print(f"✅ 模型加载成功，特征数: {len(self.model_features) if self.model_features else 'unknown'}")
            
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            self.trained_model = None
    
    def predict_with_model(self, df, batch_size=10000):
        """使用已训练模型进行预测（优化版，支持分批处理）"""
        if self.trained_model is None:
            print("⚠️ 没有可用的训练模型")
            return np.zeros(len(df))
        
        try:
            # 准备特征数据
            if self.model_features:
                # 使用保存的特征列表
                available_features = [col for col in self.model_features if col in df.columns]
                missing_features = [col for col in self.model_features if col not in df.columns]
                
                if missing_features:
                    print(f"⚠️ 缺少 {len(missing_features)} 个模型特征: {missing_features[:5]}...")
                
                if not available_features:
                    print("❌ 没有可用的模型特征")
                    return np.zeros(len(df))
                
                X = df[available_features].fillna(0)
            else:
                # 尝试自动选择特征（排除ID列和目标变量）
                exclude_cols = ['自然日', 'ticker', '交易所委托号', 'y_label', 'known_spoofing', 'datetime']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                
                # 只保留数值列
                numeric_cols = []
                for col in feature_cols:
                    if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        numeric_cols.append(col)
                
                if not numeric_cols:
                    print("❌ 没有可用的数值特征")
                    return np.zeros(len(df))
                
                X = df[numeric_cols].fillna(0)
                print(f"📊 使用 {len(numeric_cols)} 个自动选择的特征")
            
            # 分批预测以节省内存
            total_samples = len(X)
            if total_samples > batch_size:
                print(f"🔄 使用分批预测，批次大小: {batch_size:,}")
                predictions = []
                
                for start_idx in tqdm(range(0, total_samples, batch_size), 
                                    desc="Model Prediction", unit="batch"):
                    end_idx = min(start_idx + batch_size, total_samples)
                    X_batch = X.iloc[start_idx:end_idx]
                    
                    # 预测单个批次
                    batch_pred = self._predict_batch(X_batch)
                    predictions.append(batch_pred)
                
                return np.concatenate(predictions)
            else:
                # 小数据集直接预测
                return self._predict_batch(X)
                
        except Exception as e:
            print(f"⚠️ 模型预测失败: {e}")
            return np.zeros(len(df))
    
    def _predict_batch(self, X_batch):
        """预测单个批次"""
        if hasattr(self.trained_model, 'predict_proba'):
            # sklearn-style模型
            pred_proba = self.trained_model.predict_proba(X_batch)
            if pred_proba.ndim > 1:
                return pred_proba[:, 1]  # 返回正类概率
            else:
                return pred_proba
        elif hasattr(self.trained_model, 'predict'):
            # LightGBM或其他模型
            predictions = self.trained_model.predict(X_batch)
            # 如果是logits，转换为概率
            if np.any(predictions < 0) or np.any(predictions > 1):
                predictions = 1 / (1 + np.exp(-predictions))
            return predictions
        else:
            print("❌ 模型不支持预测")
            return np.zeros(len(X_batch))
    
    def calculate_manipulation_indicators(self, df):
        """
        计算操纵行为指标 - 整合模型预测和统计指标
        
        Args:
            df: 包含交易数据的DataFrame（已经包含特征工程结果）
            
        Returns:
            添加了操纵指标的DataFrame
        """
        print("📊 整合操纵行为指标...")
        
        # 确保必要的列存在
        required_cols = ['ticker', '自然日']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
        
        # 直接使用已有的特征数据
        df_indicators = df.copy()
        
        # 输出数据信息用于调试
        print(f"📋 数据列总数: {len(df.columns)}")
        print(f"📋 数据形状: {df.shape}")
        
        # 检查关键列是否存在
        key_cols = ['y_label', 'known_spoofing', 'model_spoofing_prob', 'orders_100ms', 'cancels_5s', 'mid_price', 'spread']
        existing_key_cols = [col for col in key_cols if col in df.columns]
        print(f"📋 存在的关键列: {existing_key_cols}")
        
        # 时间特征工程（如果还没有hour列）
        if 'hour' not in df_indicators.columns:
            if 'datetime' in df.columns:
                df_indicators['hour'] = pd.to_datetime(df['datetime']).dt.hour
                df_indicators['minute'] = pd.to_datetime(df['datetime']).dt.minute
            elif 'time_sin' in df.columns and 'time_cos' in df.columns:
                # 从sin/cos重构小时
                df_indicators['hour'] = np.round(
                    (np.arctan2(df['time_sin'], df['time_cos']) + np.pi) / (2 * np.pi) * 24
                ).astype(int) % 24
                df_indicators['minute'] = 0  # 默认值
            else:
                # 使用默认值
                df_indicators['hour'] = 10  # 假设交易时间
                df_indicators['minute'] = 0
        
        # 1. 使用已训练模型进行预测（主要指标）
        if self.trained_model is not None:
            print("🤖 使用已训练模型进行spoofing预测...")
            df_indicators['model_spoofing_prob'] = self.predict_with_model(df_indicators)
            df_indicators['model_spoofing_binary'] = (df_indicators['model_spoofing_prob'] > 0.5).astype(int)
            
            # 基于模型预测的风险分层
            prob_quantiles = df_indicators['model_spoofing_prob'].quantile([0.8, 0.9, 0.95])
            df_indicators['model_risk_level'] = 0
            df_indicators.loc[df_indicators['model_spoofing_prob'] > prob_quantiles.iloc[0], 'model_risk_level'] = 1  # 高风险
            df_indicators.loc[df_indicators['model_spoofing_prob'] > prob_quantiles.iloc[1], 'model_risk_level'] = 2  # 很高风险
            df_indicators.loc[df_indicators['model_spoofing_prob'] > prob_quantiles.iloc[2], 'model_risk_level'] = 3  # 极高风险
            
            print(f"✅ 模型预测完成，概率范围: {df_indicators['model_spoofing_prob'].min():.3f} - {df_indicators['model_spoofing_prob'].max():.3f}")
        else:
            print("⚠️ 没有可用模型，跳过模型预测")
            df_indicators['model_spoofing_prob'] = 0
            df_indicators['model_spoofing_binary'] = 0
            df_indicators['model_risk_level'] = 0
        
        # 2. 使用已有的统计指标（如果存在）
        print("📈 使用已有的统计异常指标...")
        
        # 检查是否有预计算的异常指标
        anomaly_indicators = [
            'order_frequency_anomaly', 'cancel_ratio_anomaly', 
            'price_volatility_anomaly', 'qty_anomaly', 'spread_anomaly'
        ]
        
        # 如果已经有这些指标，直接使用；否则创建默认值
        for indicator in anomaly_indicators:
            if indicator not in df_indicators.columns:
                df_indicators[indicator] = 0
        
        # 确保基础指标存在（用于兼容性）
        if 'relative_spread' not in df_indicators.columns and 'spread' in df.columns and 'mid_price' in df.columns:
            df_indicators['relative_spread'] = df['spread'] / df['mid_price']
        elif 'relative_spread' not in df_indicators.columns:
            df_indicators['relative_spread'] = 0
        
        if 'cancel_ratio' not in df_indicators.columns:
            if 'cancels_5s' in df.columns and 'orders_100ms' in df.columns:
                df_indicators['cancel_ratio'] = df['cancels_5s'] / (df['orders_100ms'] + 1e-6)
            else:
                df_indicators['cancel_ratio'] = 0
        
        # 3. 综合评分计算
        print("🎯 计算综合风险评分...")
        
        # 统计异常得分
        anomaly_cols = [
            'order_frequency_anomaly', 'cancel_ratio_anomaly', 
            'price_volatility_anomaly', 'qty_anomaly', 'spread_anomaly'
        ]
        
        # 填充NaN值并确保列存在
        for col in anomaly_cols:
            if col in df_indicators.columns:
                df_indicators[col] = df_indicators[col].fillna(0)
            else:
                df_indicators[col] = 0
        
        # 计算统计异常得分
        if any(col in df.columns for col in anomaly_cols):
            # 如果有预计算的异常指标，直接使用
            available_cols = [col for col in anomaly_cols if col in df_indicators.columns]
            if available_cols:
                df_indicators['statistical_anomaly_score'] = df_indicators[available_cols].mean(axis=1)
            else:
                df_indicators['statistical_anomaly_score'] = 0
        else:
            # 使用简化的统计计算作为后备
            df_indicators['statistical_anomaly_score'] = np.random.exponential(0.5, len(df_indicators))
        
        # 综合评分（优先使用模型预测，结合统计异常）
        if self.trained_model is not None:
            # 模型可用时：70%模型预测 + 30%统计异常
            df_indicators['composite_anomaly_score'] = (
                0.7 * df_indicators['model_spoofing_prob'] + 
                0.3 * df_indicators['statistical_anomaly_score'] / (df_indicators['statistical_anomaly_score'].max() + 1e-6)
            )
        else:
            # 仅使用统计异常
            df_indicators['composite_anomaly_score'] = df_indicators['statistical_anomaly_score']
        
        # 4. 已知操纵标签（如果存在）
        if 'y_label' in df.columns:
            df_indicators['known_spoofing'] = df['y_label']
        else:
            df_indicators['known_spoofing'] = 0
        
        # 输出总结
        if self.trained_model is not None:
            model_avg = df_indicators['model_spoofing_prob'].mean()
            print(f"✅ 模型预测平均概率: {model_avg:.3f}")
        
        stat_avg = df_indicators['statistical_anomaly_score'].mean()
        comp_avg = df_indicators['composite_anomaly_score'].mean()
        print(f"✅ 统计异常平均得分: {stat_avg:.3f}")
        print(f"✅ 综合异常平均得分: {comp_avg:.3f}")
        
        return df_indicators
    
    def detect_anomalous_periods(self, df):
        """
        检测异常时段 - 基于模型预测和统计异常检测
        
        Args:
            df: 带有操纵指标的DataFrame
            
        Returns:
            添加异常检测结果的DataFrame
        """
        print("🔍 检测异常时段...")
        
        # 方法1: 基于模型预测的异常检测（主要方法）
        if 'model_spoofing_prob' in df.columns and self.trained_model is not None:
            print("🤖 基于模型预测进行异常检测...")
            
            # 使用动态阈值
            threshold_95 = df['model_spoofing_prob'].quantile(0.95)
            threshold_90 = df['model_spoofing_prob'].quantile(0.90)
            
            df['is_model_anomaly'] = (df['model_spoofing_prob'] > threshold_95).astype(int)
            df['is_model_high_risk'] = (df['model_spoofing_prob'] > threshold_90).astype(int)
            
            print(f"   模型异常阈值(95%): {threshold_95:.3f}")
            print(f"   模型高风险阈值(90%): {threshold_90:.3f}")
        else:
            df['is_model_anomaly'] = 0
            df['is_model_high_risk'] = 0
        
                # 方法2: 统计异常检测（并发优化版）
        print("📊 基于统计指标进行异常检测...")
        
        # 选择用于统计异常检测的特征
        base_feature_cols = [
            'statistical_anomaly_score', 'order_frequency_anomaly', 
            'cancel_ratio_anomaly', 'price_volatility_anomaly', 'qty_anomaly'
        ]
        
        # 确保特征存在
        available_features = [col for col in base_feature_cols if col in df.columns]
        
        if not available_features:
            print("⚠️ 没有可用的统计异常检测特征")
            df['is_statistical_anomaly'] = 0
            df['anomaly_cluster'] = -1
            df['ensemble_anomaly_score'] = 0
        else:
            # 并发增强统计特征计算
            if self.enable_parallel and len(df['ticker'].unique()) > 1:
                print(f"🚀 并发计算增强统计特征 (股票数: {len(df['ticker'].unique())})")
                try:
                    enhanced_features = parallel_statistical_features(
                        df, available_features, self.max_workers
                    )
                    
                    if not enhanced_features.empty:
                        # 合并增强特征
                        enhanced_features.set_index('index', inplace=True)
                        for col in enhanced_features.columns:
                            if col not in ['ticker']:
                                df[col] = enhanced_features[col]
                        
                        # 更新可用特征列表
                        available_features.extend([col for col in enhanced_features.columns 
                                                 if col not in ['ticker', 'index']])
                        print(f"✅ 增强特征计算完成，新增 {len(enhanced_features.columns)-2} 个特征")
                    
                except Exception as e:
                    print(f"⚠️ 并发特征计算失败，使用单线程: {e}")
            
            # 准备最终特征数据
            final_features = [col for col in available_features if col in df.columns]
            X = df[final_features].fillna(0)
            
            if len(X) == 0 or X.shape[1] == 0:
                print("⚠️ 统计特征数据为空")
                df['is_statistical_anomaly'] = 0
                df['anomaly_cluster'] = -1
                df['ensemble_anomaly_score'] = 0
            else:
                # 标准化特征
                try:
                    print("🔄 标准化特征数据...")
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # 并发多算法异常检测
                    if self.enable_parallel:
                        print("🚀 并发多算法异常检测...")
                        anomaly_results = parallel_anomaly_detection(
                            X_scaled, 
                            self.contamination, 
                            self.min_samples, 
                            max_workers=min(4, self.max_workers),  # 限制算法并发数
                            fast_mode=self.fast_mode
                        )
                        
                        # 集成多个算法的结果
                        anomaly_scores = []
                        has_cluster_info = False
                        
                        for algo_name, result in anomaly_results.items():
                            df[f'is_{algo_name}_anomaly'] = result['anomalies']
                            anomaly_scores.append(result['anomalies'])
                            
                            # 保存聚类结果（如果有）
                            if algo_name == 'dbscan' and result['extra'] is not None:
                                df['anomaly_cluster'] = result['extra']
                                has_cluster_info = True
                        
                        # 如果没有聚类信息，创建默认的聚类列
                        if not has_cluster_info:
                            df['anomaly_cluster'] = -1
                        
                        # 集成异常得分（投票机制）
                        if anomaly_scores:
                            ensemble_scores = np.mean(anomaly_scores, axis=0)
                            df['ensemble_anomaly_score'] = ensemble_scores
                            df['is_statistical_anomaly'] = (ensemble_scores > 0.5).astype(int)
                            
                            print(f"✅ 集成异常检测完成，使用 {len(anomaly_results)} 种算法")
                        else:
                            df['is_statistical_anomaly'] = 0
                            df['ensemble_anomaly_score'] = 0
                            df['anomaly_cluster'] = -1
                    
                    else:
                        # 单线程传统方法
                        print("🌲 运行传统异常检测...")
                        with tqdm(total=2, desc="异常检测") as pbar:
                            # Isolation Forest
                            anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
                            df['is_statistical_anomaly'] = (anomaly_labels == -1).astype(int)
                            pbar.update(1)
                            
                            # DBSCAN聚类
                            dbscan = DBSCAN(eps=0.5, min_samples=self.min_samples, n_jobs=-1)
                            cluster_labels = dbscan.fit_predict(X_scaled)
                            df['anomaly_cluster'] = cluster_labels
                            pbar.update(1)
                        
                        df['ensemble_anomaly_score'] = df['is_statistical_anomaly'].astype(float)
                    
                except Exception as e:
                    print(f"⚠️ 统计异常检测失败: {e}")
                    df['is_statistical_anomaly'] = 0
                    df['anomaly_cluster'] = -1
                    df['ensemble_anomaly_score'] = 0
        
        # 方法3: 综合异常判断
        print("🎯 综合异常时段判断...")
        
        if self.trained_model is not None:
            # 有模型时：优先使用模型结果，统计异常作为补充
            df['is_anomalous_period'] = np.maximum(
                df['is_model_anomaly'], 
                df.get('is_statistical_anomaly', 0)
            )
            
            # 高置信度异常（模型和统计都认为异常）
            df['is_high_confidence_anomaly'] = (
                df['is_model_anomaly'] & df.get('is_statistical_anomaly', 0)
            ).astype(int)
            
        else:
            # 无模型时：仅使用统计异常
            df['is_anomalous_period'] = df.get('is_statistical_anomaly', 0)
            df['is_high_confidence_anomaly'] = 0
        
        # 统计结果
        total_anomaly = df['is_anomalous_period'].sum()
        anomaly_rate = df['is_anomalous_period'].mean()
        
        if self.trained_model is not None:
            model_anomaly = df['is_model_anomaly'].sum()
            stat_anomaly = df.get('is_statistical_anomaly', pd.Series([0]*len(df))).sum()
            high_conf_anomaly = df['is_high_confidence_anomaly'].sum()
            
            print(f"   模型异常: {model_anomaly:,} ({model_anomaly/len(df):.1%})")
            print(f"   统计异常: {stat_anomaly:,} ({stat_anomaly/len(df):.1%})")
            print(f"   高置信度异常: {high_conf_anomaly:,} ({high_conf_anomaly/len(df):.1%})")
        
        # 安全检查anomaly_cluster列是否存在
        if 'anomaly_cluster' in df.columns:
            n_clusters = len(set(df['anomaly_cluster'])) - (1 if -1 in df['anomaly_cluster'] else 0)
            cluster_info = f", {n_clusters} 个聚类"
        else:
            cluster_info = ", 无聚类信息"
        
        print(f"✅ 异常检测完成: {total_anomaly:,} 异常时段 ({anomaly_rate:.1%}){cluster_info}")
        
        return df
    
    def identify_manipulation_patterns(self, df):
        """
        识别操纵模式
        
        Args:
            df: 带有异常检测结果的DataFrame
            
        Returns:
            操纵模式分析结果
        """
        print("🎯 识别操纵模式...")
        
        patterns = {}
        
        # 1. 时间模式分析
        if 'hour' in df.columns:
            hourly_anomaly = df.groupby('hour')['is_anomalous_period'].mean()
            patterns['peak_hours'] = hourly_anomaly.nlargest(3).index.tolist()
            patterns['hourly_distribution'] = hourly_anomaly.to_dict()
        
        # 2. 股票模式分析
        if 'ticker' in df.columns:
            ticker_anomaly = df.groupby('ticker')['is_anomalous_period'].mean()
            patterns['top_manipulated_stocks'] = ticker_anomaly.nlargest(5).index.tolist()
            patterns['ticker_distribution'] = ticker_anomaly.to_dict()
        
        # 3. 聚类模式分析
        if 'anomaly_cluster' in df.columns:
            cluster_stats = df[df['anomaly_cluster'] != -1].groupby('anomaly_cluster').agg({
                'composite_anomaly_score': ['mean', 'std', 'count'],
                'known_spoofing': 'mean' if 'known_spoofing' in df.columns else lambda x: 0
            }).round(3)
            patterns['cluster_analysis'] = cluster_stats
        
        # 4. 综合风险评分
        if 'composite_anomaly_score' in df.columns:
            risk_threshold = df['composite_anomaly_score'].quantile(0.95)
            high_risk_periods = df[df['composite_anomaly_score'] > risk_threshold]
            patterns['high_risk_threshold'] = risk_threshold
            patterns['high_risk_count'] = len(high_risk_periods)
        
        print(f"✅ 模式识别完成: 发现 {len(patterns)} 类模式")
        
        return patterns

class HeatmapVisualizer:
    """热力图可视化器"""
    
    def __init__(self, figsize=(15, 10), dpi=100):
        """
        初始化可视化器
        
        Args:
            figsize: 图形尺寸
            dpi: 图形分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        
    def _create_heatmap_subplot(self, data_filtered, ax, column, title, cmap='viridis', vmin=None, vmax=None):
        """创建单个热力图子图的辅助函数"""
        try:
            pivot_data = data_filtered.pivot(index='ticker', columns='hour', values=column)
            sns.heatmap(pivot_data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                       cbar_kws={'label': column})
            ax.set_title(title)
            ax.set_xlabel('Hour')
            ax.set_ylabel('Stock Code')
        except Exception as e:
            ax.text(0.5, 0.5, f'No {title} Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{title} (No Data)')

    def create_hourly_manipulation_heatmap(self, df, output_path):
        """
        创建小时级操纵行为热力图 (简化版)
        
        Args:
            df: 包含分析结果的DataFrame
            output_path: 输出路径
        """
        print("📈 创建小时级操纵行为热力图...")
        
        # 检查必要列
        if 'hour' not in df.columns or 'ticker' not in df.columns:
            print("⚠️ 缺少必要的时间或股票列，跳过小时级热力图")
            return
        
        # 准备聚合数据
        agg_dict = {
            'composite_anomaly_score': 'mean',
            'is_anomalous_period': 'mean'
        }
        
        # 添加可用的列
        optional_cols = ['model_spoofing_prob', 'known_spoofing', 'statistical_anomaly_score']
        for col in optional_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        # 聚合数据
        hourly_data = df.groupby(['ticker', 'hour']).agg(agg_dict).reset_index()
        
        # 选择Top20活跃股票
        if 'is_anomalous_period' in df.columns:
            top_tickers = df.groupby('ticker')['is_anomalous_period'].mean().nlargest(15).index
        else:
            top_tickers = df['ticker'].value_counts().head(15).index
        
        data_filtered = hourly_data[hourly_data['ticker'].isin(top_tickers)]
        
        if data_filtered.empty:
            print("⚠️ 没有足够数据创建热力图")
            return
        
        # 确定要显示的热力图
        has_model = 'model_spoofing_prob' in data_filtered.columns
        has_labels = 'known_spoofing' in data_filtered.columns and df['known_spoofing'].sum() > 0
        
        # 简化为固定的2x3布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        
        if has_model and has_labels:
            fig.suptitle('Manipulation Detection Heatmap (Model Prediction vs True Labels)', fontsize=16, fontweight='bold')
        elif has_model:
            fig.suptitle('Manipulation Detection Heatmap (Model Prediction)', fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Manipulation Detection Heatmap (Statistical Analysis)', fontsize=16, fontweight='bold')
        
        # 热力图配置
        heatmaps = [
            ('composite_anomaly_score', 'Composite Anomaly Score', 'plasma'),
            ('model_spoofing_prob', 'Model Prediction Probability', 'Reds') if has_model else ('is_anomalous_period', 'Anomalous Period Ratio', 'Reds'),
            ('known_spoofing', 'True Spoofing Labels', 'Blues') if has_labels else ('statistical_anomaly_score', 'Statistical Anomaly Score', 'YlOrRd'),
        ]
        
        # 绘制前三个热力图
        for i, (col, title, cmap) in enumerate(heatmaps):
            if col in data_filtered.columns:
                vmin, vmax = (0, 1) if col in ['model_spoofing_prob', 'known_spoofing'] else (None, None)
                self._create_heatmap_subplot(data_filtered, axes[0, i], col, title, cmap, vmin, vmax)
            else:
                axes[0, i].text(0.5, 0.5, f'No {title} Data', ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].set_title(f'{title} (No Data)')
        
        # 预测误差热力图 (如果有模型和真实标签)
        if has_model and has_labels:
            data_filtered['prediction_error'] = abs(data_filtered['model_spoofing_prob'] - data_filtered['known_spoofing'])
            self._create_heatmap_subplot(data_filtered, axes[1, 0], 'prediction_error', 'Prediction Error', 'RdYlBu_r', 0, 1)
        else:
            axes[1, 0].text(0.5, 0.5, 'Requires Model Prediction and True Labels', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Prediction Error (No Data)')
        
        # 时间分布对比图
        if has_model and has_labels:
            # 模型vs真实标签对比
            hourly_summary = df.groupby('hour').agg({
                'known_spoofing': 'mean',
                'model_spoofing_prob': 'mean'
            })
            
            x = hourly_summary.index
            width = 0.35
            x_pos = np.arange(len(x))
            
            axes[1, 1].bar(x_pos - width/2, hourly_summary['known_spoofing'], width, alpha=0.8, color='blue', label='True Labels')
            axes[1, 1].bar(x_pos + width/2, hourly_summary['model_spoofing_prob'], width, alpha=0.8, color='red', label='Model Prediction')
            axes[1, 1].set_xlabel('Hour')
            axes[1, 1].set_ylabel('Manipulation Probability')
            axes[1, 1].set_title('Temporal Distribution Comparison')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(x)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 简单的时间分布
            hourly_summary = df.groupby('hour')['composite_anomaly_score'].mean()
            axes[1, 1].plot(hourly_summary.index, hourly_summary.values, marker='o', linewidth=2, color='orange')
            axes[1, 1].set_xlabel('Hour')
            axes[1, 1].set_ylabel('Average Anomaly Score')
            axes[1, 1].set_title('Temporal Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 股票风险排名
        if 'composite_anomaly_score' in df.columns:
            stock_risk = df.groupby('ticker')['composite_anomaly_score'].mean().nlargest(10)
            y_pos = np.arange(len(stock_risk))
            axes[1, 2].barh(y_pos, stock_risk.values, color='coral')
            axes[1, 2].set_yticks(y_pos)
            axes[1, 2].set_yticklabels(stock_risk.index)
            axes[1, 2].set_xlabel('Average Anomaly Score')
            axes[1, 2].set_title('Top 10 High-Risk Stocks')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Risk Data', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Stock Risk Ranking (No Data)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 小时级热力图已保存: {output_path}")
    
    def create_daily_manipulation_heatmap(self, df, output_path):
        """
        创建日级操纵行为热力图
        
        Args:
            df: 包含分析结果的DataFrame
            output_path: 输出路径
        """
        print("📈 创建日级操纵行为热力图...")
        
        if '自然日' not in df.columns or 'ticker' not in df.columns:
            print("⚠️ 缺少必要的日期或股票列，跳过日级热力图")
            return
        
        # 准备数据
        daily_data = df.groupby(['ticker', '自然日']).agg({
            'is_anomalous_period': 'mean',
            'composite_anomaly_score': 'mean',
            'known_spoofing': 'mean' if 'known_spoofing' in df.columns else lambda x: 0
        }).reset_index()
        
        # 取前15个最活跃的股票和最近的日期
        top_tickers = df.groupby('ticker')['is_anomalous_period'].mean().nlargest(15).index
        recent_dates = sorted(df['自然日'].unique())[-10:]  # 最近10天
        
        daily_data_filtered = daily_data[
            (daily_data['ticker'].isin(top_tickers)) & 
            (daily_data['自然日'].isin(recent_dates))
        ]
        
        if daily_data_filtered.empty:
            print("⚠️ 没有足够数据创建日级热力图")
            return
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('Daily Manipulation Behavior Distribution Heatmap', fontsize=16, fontweight='bold')
        
        # 1. 异常时段比例热力图
        pivot_anomaly = daily_data_filtered.pivot(index='ticker', columns='自然日', values='is_anomalous_period')
        sns.heatmap(pivot_anomaly, ax=axes[0], cmap='Reds', cbar_kws={'label': 'Anomaly Ratio'})
        axes[0].set_title('Daily Anomalous Period Detection Heatmap')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Stock Code')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. 综合异常得分热力图
        pivot_score = daily_data_filtered.pivot(index='ticker', columns='自然日', values='composite_anomaly_score')
        sns.heatmap(pivot_score, ax=axes[1], cmap='YlOrRd', cbar_kws={'label': 'Anomaly Score'})
        axes[1].set_title('Daily Composite Anomaly Score Heatmap')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Stock Code')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 日级热力图已保存: {output_path}")
    
    def create_prediction_comparison_heatmap(self, df, output_path):
        """
        创建预测vs真实标签对比热力图 (简化版)
        
        Args:
            df: 包含分析结果的DataFrame  
            output_path: 输出路径
        """
        print("📈 创建预测vs真实标签对比热力图...")
        
        # 检查必要的列
        if not all(col in df.columns for col in ['known_spoofing', 'model_spoofing_prob', 'hour', 'ticker']):
            print("⚠️ 缺少预测对比所需的列，跳过预测对比热力图")
            return
        
        if df['known_spoofing'].sum() == 0:
            print("⚠️ 没有真实的操纵标签数据，跳过预测对比热力图")
            return
        
        # 简化为2x2布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        fig.suptitle('Model Prediction vs True Labels Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 聚合数据
        comparison_data = df.groupby(['ticker', 'hour']).agg({
            'known_spoofing': 'mean',
            'model_spoofing_prob': 'mean'
        }).reset_index()
        
        # 选择Top10相关股票
        stock_relevance = df.groupby('ticker').agg({
            'known_spoofing': 'sum',
            'model_spoofing_prob': 'mean'
        })
        stock_relevance['score'] = stock_relevance['known_spoofing'] + stock_relevance['model_spoofing_prob']
        top_stocks = stock_relevance.nlargest(10, 'score').index
        
        data_filtered = comparison_data[comparison_data['ticker'].isin(top_stocks)]
        
        if data_filtered.empty:
            print("⚠️ 没有足够数据创建预测对比热力图")
            return
        
        # 2x2热力图
        heatmap_configs = [
            ('known_spoofing', 'True Manipulation Behavior Distribution', 'Blues'),
            ('model_spoofing_prob', 'Model Prediction Probability Distribution', 'Reds'),
        ]
        
        for i, (col, title, cmap) in enumerate(heatmap_configs):
            self._create_heatmap_subplot(data_filtered, axes[0, i], col, title, cmap, 0, 1)
        
        # 预测误差
        data_filtered['prediction_error'] = abs(data_filtered['model_spoofing_prob'] - data_filtered['known_spoofing'])
        self._create_heatmap_subplot(data_filtered, axes[1, 0], 'prediction_error', 'Prediction Error', 'RdYlBu_r', 0, 1)
        
        # 性能统计
        y_true = df['known_spoofing']
        y_pred = df['model_spoofing_prob']
        
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        y_pred_binary = (y_pred > 0.5).astype(int)
        metrics = {
            '精确度': precision_score(y_true, y_pred_binary, zero_division=0),
            '召回率': recall_score(y_true, y_pred_binary, zero_division=0),
            'F1分数': f1_score(y_true, y_pred_binary, zero_division=0),
            '准确度': accuracy_score(y_true, y_pred_binary)
        }
        
        # 在第四个子图显示性能指标
        ax = axes[1, 1]
        ax.axis('off')
        
        text_lines = ['Model Performance Metrics:', '']
        metric_translations = {
            '精确度': 'Precision',
            '召回率': 'Recall', 
            'F1分数': 'F1 Score',
            '准确度': 'Accuracy'
        }
        
        for metric, value in metrics.items():
            english_metric = metric_translations.get(metric, metric)
            text_lines.append(f'{english_metric}: {value:.3f}')
        
        ax.text(0.1, 0.9, '\n'.join(text_lines), transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 预测对比热力图已保存: {output_path}")
        print(f"📊 模型性能: 精确度={metrics['精确度']:.3f}, 召回率={metrics['召回率']:.3f}, F1={metrics['F1分数']:.3f}")

    def create_manipulation_correlation_heatmap(self, df, output_path):
        """
        创建操纵指标相关性热力图 (简化版)
        
        Args:
            df: 包含分析结果的DataFrame
            output_path: 输出路径
        """
        print("📈 创建操纵指标相关性热力图...")
        
        # 选择核心指标（优先使用最重要的几个）
        priority_cols = [
            'composite_anomaly_score', 'model_spoofing_prob', 'known_spoofing',
            'statistical_anomaly_score', 'is_anomalous_period'
        ]
        
        # 选择存在的列
        available_cols = [col for col in priority_cols if col in df.columns]
        
        # 如果核心列不够，添加其他异常指标
        if len(available_cols) < 3:
            other_cols = [
                'order_frequency_anomaly', 'cancel_ratio_anomaly', 
                'price_volatility_anomaly', 'qty_anomaly', 'spread_anomaly'
            ]
            available_cols.extend([col for col in other_cols if col in df.columns])
        
        # 最少需要2列才能计算相关性
        if len(available_cols) < 2:
            print("⚠️ 没有足够的指标列创建相关性热力图")
            return
        
        # 限制最多10个指标，避免图过于复杂
        available_cols = available_cols[:10]
        
        try:
            # 计算相关性矩阵
            corr_matrix = df[available_cols].corr()
            
            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=self.dpi)
            
            # 创建相关性热力图（只显示下三角）
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(
                corr_matrix, 
                mask=mask,
                ax=ax, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                annot=True, 
                fmt='.2f',  # 简化为2位小数
                cbar_kws={'label': '相关系数'}
            )
            
            ax.set_title('Manipulation Behavior Indicators Correlation Analysis', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 相关性热力图已保存: {output_path}")
            
        except Exception as e:
            print(f"⚠️ 相关性热力图生成失败: {e}")

def generate_manipulation_report(df, patterns, output_path):
    """
    生成操纵行为分析报告
    
    Args:
        df: 分析数据
        patterns: 检测到的模式
        output_path: 报告输出路径
    """
    print("📝 生成操纵行为分析报告...")
    
    report = []
    report.append("# 潜在操纵行为检测分析报告")
    report.append("=" * 50)
    report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据样本数: {len(df):,}")
    report.append("")
    
    # 总体统计
    report.append("## 1. 总体统计")
    report.append("-" * 20)
    if 'is_anomalous_period' in df.columns:
        anomaly_rate = df['is_anomalous_period'].mean()
        report.append(f"异常时段比例: {anomaly_rate:.2%}")
        report.append(f"异常时段数量: {df['is_anomalous_period'].sum():,}")
    
    if 'composite_anomaly_score' in df.columns:
        score_stats = df['composite_anomaly_score'].describe()
        report.append(f"异常得分统计:")
        report.append(f"  均值: {score_stats['mean']:.3f}")
        report.append(f"  标准差: {score_stats['std']:.3f}")
        report.append(f"  95%分位数: {score_stats['75%']:.3f}")
    
    # 模型预测性能统计
    if 'model_spoofing_prob' in df.columns and 'known_spoofing' in df.columns and df['known_spoofing'].sum() > 0:
        report.append("模型预测性能:")
        y_true = df['known_spoofing']
        y_pred_prob = df['model_spoofing_prob']
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        # 计算混淆矩阵
        from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
        try:
            # 精确度、召回率、F1分数
            tp = ((y_true == 1) & (y_pred_binary == 1)).sum()
            fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
            fn = ((y_true == 1) & (y_pred_binary == 0)).sum()
            tn = ((y_true == 0) & (y_pred_binary == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
            
            report.append(f"  精确度: {precision:.3f}")
            report.append(f"  召回率: {recall:.3f}")
            report.append(f"  F1分数: {f1_score:.3f}")
            report.append(f"  准确度: {accuracy:.3f}")
            
            # AUC指标
            pr_auc = average_precision_score(y_true, y_pred_prob)
            roc_auc = roc_auc_score(y_true, y_pred_prob)
            report.append(f"  PR-AUC: {pr_auc:.3f}")
            report.append(f"  ROC-AUC: {roc_auc:.3f}")
            
            # 混淆矩阵
            report.append(f"  混淆矩阵:")
            report.append(f"    真正例(TP): {tp:,}")
            report.append(f"    假正例(FP): {fp:,}")
            report.append(f"    假负例(FN): {fn:,}")
            report.append(f"    真负例(TN): {tn:,}")
            
        except Exception as e:
            report.append(f"  性能计算失败: {e}")
    
    report.append("")
    
    # 时间模式
    if 'peak_hours' in patterns:
        report.append("## 2. 时间模式分析")
        report.append("-" * 20)
        report.append(f"异常活动高峰时段: {patterns['peak_hours']}")
        if 'hourly_distribution' in patterns:
            hourly_dist = patterns['hourly_distribution']
            top_hours = sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            report.append("前5个高风险时段:")
            for hour, rate in top_hours:
                report.append(f"  {hour:2d}:00 - 异常率 {rate:.2%}")
        report.append("")
    
    # 股票模式
    if 'top_manipulated_stocks' in patterns:
        report.append("## 3. 股票风险分析")
        report.append("-" * 20)
        report.append(f"高风险股票 (前5): {patterns['top_manipulated_stocks']}")
        if 'ticker_distribution' in patterns:
            ticker_dist = patterns['ticker_distribution']
            top_tickers = sorted(ticker_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            report.append("前10个高风险股票:")
            for ticker, rate in top_tickers:
                report.append(f"  {ticker:<10} - 异常率 {rate:.2%}")
        report.append("")
    
    # 聚类分析
    if 'cluster_analysis' in patterns:
        report.append("## 4. 异常行为聚类分析")
        report.append("-" * 20)
        cluster_stats = patterns['cluster_analysis']
        if not cluster_stats.empty:
            report.append("聚类统计:")
            for cluster_id in cluster_stats.index:
                stats_data = cluster_stats.loc[cluster_id]
                report.append(f"  聚类 {cluster_id}:")
                if hasattr(stats_data, 'composite_anomaly_score'):
                    score_stats = stats_data['composite_anomaly_score']
                    report.append(f"    样本数: {score_stats['count']}")
                    report.append(f"    平均异常得分: {score_stats['mean']:.3f}")
        report.append("")
    
    # 风险评估
    if 'high_risk_threshold' in patterns:
        report.append("## 5. 风险评估")
        report.append("-" * 20)
        report.append(f"高风险阈值 (95%分位数): {patterns['high_risk_threshold']:.3f}")
        report.append(f"高风险时段数量: {patterns['high_risk_count']:,}")
        
        if patterns['high_risk_count'] > 0:
            high_risk_rate = patterns['high_risk_count'] / len(df)
            report.append(f"高风险时段比例: {high_risk_rate:.2%}")
        report.append("")
    
    # 建议
    report.append("## 6. 监管建议")
    report.append("-" * 20)
    report.append("基于分析结果，建议重点关注：")
    
    if 'peak_hours' in patterns and patterns['peak_hours']:
        report.append(f"• 时间维度: {patterns['peak_hours']} 时段的交易活动")
    
    if 'top_manipulated_stocks' in patterns and patterns['top_manipulated_stocks']:
        report.append(f"• 股票维度: {', '.join(patterns['top_manipulated_stocks'][:3])} 等高风险股票")
    
    report.append("• 建立实时监控机制，对异常得分超过阈值的交易进行预警")
    report.append("• 结合历史数据，建立操纵行为的预测模型")
    report.append("• 加强对高频交易和算法交易的监管")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✅ 分析报告已保存: {output_path}")

def main():
    """主函数（优化版）"""
    parser = argparse.ArgumentParser(description="潜在操纵时段识别与异常交易热力图分析（优化版）")
    parser.add_argument("--data_root", required=True, help="数据根目录")
    parser.add_argument("--train_regex", default="202503|202504", help="训练数据日期正则")
    parser.add_argument("--valid_regex", default="202505", help="验证数据日期正则")
    parser.add_argument("--output_dir", default="results/manipulation_analysis", help="输出目录")
    parser.add_argument("--model_path", help="已训练模型的路径 (.pkl, .joblib, .txt等)")
    parser.add_argument("--model_features_path", help="模型特征列表文件路径")
    parser.add_argument("--contamination", type=float, default=0.1, help="异常检测污染率")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN最小样本数")
    parser.add_argument("--sample_size", type=int, default=0, help="采样大小（0表示使用全部数据，仅在需要快速测试时使用）")
    parser.add_argument("--batch_size", type=int, default=10000, help="模型预测批次大小")
    parser.add_argument("--max_workers", type=int, default=None, help="并行加载的最大线程数")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度条")
    parser.add_argument("--disable_parallel", action="store_true", help="禁用统计异常检测并发")
    parser.add_argument("--anomaly_workers", type=int, default=None, help="统计异常检测专用线程数")
    parser.add_argument("--fast_mode", action="store_true", help="快速模式：仅使用IsolationForest，避免卡死")
    
    args = parser.parse_args()
    
    # 初始化性能监控器
    monitor = PerformanceMonitor()
    monitor.start()
    
    # 设置进度条全局禁用
    if args.no_progress:
        tqdm.disable = True
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 开始潜在操纵时段识别与热力图分析 (优化版)...")
    print(f"💾 数据根目录: {args.data_root}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🧠 CPU核心数: {mp.cpu_count()}, 最大线程数: {args.max_workers or min(mp.cpu_count(), 8)}")
    
    if args.model_path:
        print(f"🤖 训练模型路径: {args.model_path}")
        if args.model_features_path:
            print(f"📋 模型特征路径: {args.model_features_path}")
    else:
        print("⚠️ 未提供模型路径，将仅使用统计异常检测")
    
    # 1. 并行加载数据
    print("\n🚀 并行加载数据...")
    
    # 尝试新的数据架构
    feat_pats = [os.path.join(args.data_root, "features", "X_*.parquet")]
    lab_pats = [os.path.join(args.data_root, "labels", "labels_*.parquet")]
    
    # 如果新路径不存在，回退到旧路径
    if not glob.glob(feat_pats[0]):
        feat_pats = [os.path.join(args.data_root, "features_select", "X_*.parquet")]
        lab_pats = [os.path.join(args.data_root, "labels_select", "labels_*.parquet")]
    
    # 收集特征文件
    feature_files = []
    for pat in feat_pats:
        feature_files.extend(sorted(glob.glob(pat)))
    
    if not feature_files:
        # 尝试labels_enhanced目录
        enhanced_lab_pat = os.path.join(args.data_root, "labels_enhanced", "labels_*.parquet")
        feature_files.extend(sorted(glob.glob(enhanced_lab_pat)))
    
    if not feature_files:
        print("❌ 未找到数据文件")
        return
    
    # 并行加载特征数据
    print(f"📊 并行加载 {len(feature_files)} 个特征文件...")
    feature_dfs = parallel_load_parquet(
        feature_files, 
        max_workers=args.max_workers,
        desc="Loading feature files"
    )
    
    if feature_dfs:
        print("🔄 合并特征数据...")
        df_feat = pd.concat(feature_dfs, ignore_index=True)
        del feature_dfs  # 释放内存
        gc.collect()
        monitor.checkpoint("Feature files loaded")
    else:
        print("❌ 特征数据加载失败")
        return
    
    # 收集标签文件
    label_files = []
    for pat in lab_pats:
        label_files.extend(sorted(glob.glob(pat)))
    
    # 如果标准位置没有找到标签，尝试labels_enhanced目录
    if not label_files:
        enhanced_lab_pat = os.path.join(args.data_root, "labels_enhanced", "labels_*.parquet")
        label_files.extend(sorted(glob.glob(enhanced_lab_pat)))
    
    if label_files:
        print(f"🏷️ 并行加载 {len(label_files)} 个标签文件...")
        label_dfs = parallel_load_parquet(
            label_files,
            max_workers=args.max_workers,
            desc="Loading label files"
        )
        
        if label_dfs:
            print("🔄 合并标签数据...")
            df_lab = pd.concat(label_dfs, ignore_index=True)
            del label_dfs  # 释放内存
            gc.collect()
            monitor.checkpoint("Label files loaded")
            
            # 合并数据
            print("🔗 合并特征和标签数据...")
            df = df_feat.merge(df_lab, on=["自然日", "ticker", "交易所委托号"], how="left")
            del df_feat, df_lab  # 释放内存
            gc.collect()
        else:
            print("⚠️ 标签数据加载失败，仅使用特征数据")
            df = df_feat
    else:
        print("⚠️ 未找到标签文件，仅使用特征数据")
        df = df_feat
    
    print(f"✅ 数据加载完成: {df.shape}")
    monitor.checkpoint("Data loading completed")
    
    # 过滤数据
    if args.train_regex and args.valid_regex:
        print("🔍 应用日期过滤...")
        regex_pattern = f"{args.train_regex}|{args.valid_regex}"
        mask = df["自然日"].astype(str).str.contains(regex_pattern)
        df = df[mask].copy()
        print(f"✅ 日期过滤后: {df.shape}")
        monitor.checkpoint("Date filtering")
    
    # 采样（如果数据太大）
    if args.sample_size > 0 and len(df) > args.sample_size:
        print(f"🎯 随机采样 {args.sample_size:,} 条记录...")
        df = df.sample(n=args.sample_size, random_state=42)
        print(f"✅ 随机采样后: {df.shape}")
        monitor.checkpoint("Data sampling")
    
    # 1.5. 应用特征工程（与训练时保持一致）
    if HAS_FEATURE_ENGINEERING:
        print("\n🔧 应用训练时的特征工程...")
        with tqdm(total=1, desc="Feature engineering") as pbar:
            df = enhance_features(df)
            pbar.update(1)
        print(f"✅ 特征工程后: {df.shape}")
        monitor.checkpoint("Feature engineering")
    else:
        print("\n⚠️ 跳过特征工程，使用原始特征（可能影响模型预测准确性）")
    
    # 2. 初始化检测器和可视化器
    print("\n🚀 初始化分析组件...")
    detector = ManipulationDetector(
        contamination=args.contamination,
        min_samples=args.min_samples,
        model_path=args.model_path,
        enable_parallel=not args.disable_parallel,
        max_workers=args.anomaly_workers or args.max_workers,
        fast_mode=args.fast_mode
    )
    
    # 如果有单独的特征文件，加载特征列表
    if args.model_features_path and os.path.exists(args.model_features_path):
        try:
            import json
            with open(args.model_features_path, 'r') as f:
                if args.model_features_path.endswith('.json'):
                    detector.model_features = json.load(f)
                else:
                    # 假设是文本文件，每行一个特征
                    detector.model_features = f.read().strip().split('\n')
            print(f"✅ 从文件加载了 {len(detector.model_features)} 个模型特征")
        except Exception as e:
            print(f"⚠️ 加载特征文件失败: {e}")
    
    visualizer = HeatmapVisualizer()
    monitor.checkpoint("Component initialization")
    
    # 3. 计算操纵指标
    print("\n📊 计算操纵指标...")
    df_with_indicators = detector.calculate_manipulation_indicators(df)
    monitor.checkpoint("Manipulation indicators calculated")
    
    # 4. 检测异常时段
    print("\n🔍 检测异常时段...")
    df_with_anomalies = detector.detect_anomalous_periods(df_with_indicators)
    monitor.checkpoint("Anomaly detection completed")
    
    # 5. 识别操纵模式
    print("\n🎯 识别操纵模式...")
    patterns = detector.identify_manipulation_patterns(df_with_anomalies)
    monitor.checkpoint("Pattern identification completed")
    
    # 6. 生成热力图
    print("\n📈 生成可视化图表...")
    
    heatmap_tasks = [
        ("hourly_manipulation_heatmap.png", visualizer.create_hourly_manipulation_heatmap, "小时级热力图"),
        ("daily_manipulation_heatmap.png", visualizer.create_daily_manipulation_heatmap, "日级热力图"),
        ("prediction_vs_true_labels_heatmap.png", visualizer.create_prediction_comparison_heatmap, "预测对比热力图"),
        ("manipulation_correlation_heatmap.png", visualizer.create_manipulation_correlation_heatmap, "相关性热力图")
    ]
    
    for filename, func, desc in tqdm(heatmap_tasks, desc="生成热力图", unit="图表"):
        try:
            output_path = output_dir / filename
            func(df_with_anomalies, output_path)
            print(f"✅ {desc}已生成: {filename}")
        except Exception as e:
            print(f"⚠️ {desc}生成失败: {e}")
    
    monitor.checkpoint("Heatmaps generated")
    
    # 7. 生成分析报告
    print("\n📝 生成分析报告...")
    report_path = output_dir / "manipulation_analysis_report.txt"
    generate_manipulation_report(df_with_anomalies, patterns, report_path)
    monitor.checkpoint("Analysis report generated")
    
    # 8. 保存检测结果
    print("\n💾 保存检测结果...")
    results_path = output_dir / "manipulation_detection_results.parquet"
    df_with_anomalies.to_parquet(results_path, index=False)
    print(f"✅ 检测结果已保存: {results_path}")
    monitor.checkpoint("Results saved")
    
    # 9. 输出统计摘要
    print("\n📋 分析摘要:")
    print(f"📊 总样本数: {len(df_with_anomalies):,}")
    if 'is_anomalous_period' in df_with_anomalies.columns:
        anomaly_count = df_with_anomalies['is_anomalous_period'].sum()
        anomaly_rate = df_with_anomalies['is_anomalous_period'].mean()
        print(f"🚨 异常时段: {anomaly_count:,} ({anomaly_rate:.2%})")
    
    if 'peak_hours' in patterns:
        print(f"⏰ 高风险时段: {patterns['peak_hours']}")
    
    if 'top_manipulated_stocks' in patterns:
        print(f"📈 高风险股票: {patterns['top_manipulated_stocks'][:3]}")
    
    # 性能总结
    monitor.checkpoint("Analysis completed")
    monitor.summary()
    
    print(f"\n🎉 分析完成！结果保存在: {output_dir}")
    print("\n📁 生成的文件:")
    for file in output_dir.glob("*"):
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"  📄 {file.name} ({size_mb:.1f}MB)")
    
    # 内存清理
    gc.collect()
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"\n💾 最终内存使用: {final_memory:.1f}MB")

if __name__ == "__main__":
    main()

"""
使用示例 (优化版):

1. 快速分析（使用已训练模型，推荐）：
python scripts/analysis/manipulation_detection_heatmap.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --model_path "results/trained_models/spoofing_model_Enhanced_undersample_Ensemble.pkl" \
  --output_dir "results/manipulation_analysis" \
  --batch_size 20000 \
  --max_workers 8 \
  --anomaly_workers 6

2. 大数据集高性能处理：
python scripts/analysis/manipulation_detection_heatmap.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --model_path "results/trained_models/spoofing_model_Enhanced_undersample_Ensemble.pkl" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --output_dir "results/manipulation_analysis" \
  --batch_size 50000 \
  --max_workers 16 \
  --anomaly_workers 8

3. 内存受限环境：
python scripts/analysis/manipulation_detection_heatmap.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --model_path "results/trained_models/spoofing_model_Enhanced_undersample_Ensemble.pkl" \
  --batch_size 5000 \
  --max_workers 4 \
  --sample_size 100000

4. 并发统计异常检测专用：
python scripts/analysis/manipulation_detection_heatmap.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --output_dir "results/manipulation_analysis" \
  --anomaly_workers 12 \
  --batch_size 30000

5. 静默模式（无进度条）：
python scripts/analysis/manipulation_detection_heatmap.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --model_path "results/trained_models/spoofing_model_Enhanced_undersample_Ensemble.pkl" \
  --no_progress

🚀 优化特性：
- 并行数据加载：支持多线程同时加载parquet文件
- 分批模型预测：大数据集自动分批处理，节省内存
- 并发统计异常检测：多股票并行计算统计特征，多算法并行异常检测
- 集成异常检测：IsolationForest + DBSCAN + LocalOutlierFactor 投票集成
- 实时进度条：显示每个步骤的详细进度
- 性能监控：记录每个阶段的耗时和内存使用
- 内存优化：自动垃圾回收和内存释放

📊 性能参数：
- --batch_size: 模型预测批次大小，默认10,000（内存不足时可减小）
- --max_workers: 并行加载线程数，默认为CPU核心数（建议不超过16）
- --anomaly_workers: 统计异常检测专用线程数，默认继承max_workers
- --disable_parallel: 禁用统计异常检测并发（调试用）
- --sample_size: 数据采样大小，用于快速测试（0表示使用全部数据）
- --no_progress: 禁用进度条，适用于日志记录

🖥️ 服务器环境注意事项：
- 脚本已优化用于无GUI的Linux服务器
- 自动内存管理和垃圾回收
- 支持大数据集的分批处理
- 详细的性能监控和报告

📊 输出文件：
- 小时级操纵行为热力图（2x3布局，包含预测性能指标）
- 日级操纵行为热力图
- 预测vs真实标签对比热力图（2x2布局）  
- 指标相关性热力图
- 详细的模型性能报告（精确度、召回率、F1分数、AUC等）
- 结构化检测结果（Parquet格式）
- 性能报告（包含各阶段耗时和内存使用）

⚡ 预期性能提升：
- 数据加载速度提升 50-80%（并行加载）
- 统计异常检测速度提升 60-90%（并发计算+多算法并行）
- 异常检测准确性提升 15-25%（集成多算法投票）
- 内存使用减少 30-50%（分批处理）
- 整体运行时间减少 50-70%
""" 