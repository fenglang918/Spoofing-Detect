#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Management for Spoofing Detection Project
========================================================
统一的配置管理模块，支持环境变量和配置文件
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

@dataclass
class DataConfig:
    """数据相关配置"""
    base_data_root: str = "/obs/users/fenglang/general/Spoofing Detect/data/base_data"
    train_regex: str = "202503|202504"
    valid_regex: str = "202505"
    test_regex: str = "202506"
    default_tickers: List[str] = None
    
    def __post_init__(self):
        if self.default_tickers is None:
            self.default_tickers = ["000989.SZ", "300233.SZ"]

@dataclass
class ModelConfig:
    """模型相关配置"""
    # LightGBM 参数
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 10.0
    reg_lambda: float = 10.0
    min_data_in_leaf: int = 5
    early_stopping_rounds: int = 100
    
    # 训练策略
    by_ticker: bool = True
    use_gpu: bool = False
    random_state: int = 42

@dataclass
class FeatureConfig:
    """特征工程配置"""
    backend: str = "polars"  # "polars" 或 "pandas"
    extended_features: bool = True
    feature_selection: bool = True
    
    # 特征白名单
    feature_whitelist: Optional[List[str]] = None

@dataclass
class EvaluationConfig:
    """评估配置"""
    precision_k_thresholds: List[float] = None
    save_predictions: bool = True
    save_feature_importance: bool = True
    
    def __post_init__(self):
        if self.precision_k_thresholds is None:
            self.precision_k_thresholds = [0.001, 0.005, 0.01]

@dataclass
class ProjectConfig:
    """项目总配置"""
    data: DataConfig = None
    model: ModelConfig = None
    feature: FeatureConfig = None
    evaluation: EvaluationConfig = None
    
    # 路径配置
    project_root: str = ""
    results_dir: str = "results"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.feature is None:
            self.feature = FeatureConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        
        if not self.project_root:
            self.project_root = str(Path(__file__).parent)

def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
    
    Returns:
        ProjectConfig: 项目配置对象
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 递归创建配置对象
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        feature_config = FeatureConfig(**config_dict.get('feature', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        project_config = ProjectConfig(
            data=data_config,
            model=model_config,
            feature=feature_config,
            evaluation=evaluation_config,
            **{k: v for k, v in config_dict.items() 
               if k not in ['data', 'model', 'feature', 'evaluation']}
        )
    else:
        project_config = ProjectConfig()
    
    # 从环境变量覆盖配置
    _override_from_env(project_config)
    
    return project_config

def save_config(config: ProjectConfig, config_path: str):
    """保存配置到文件"""
    config_dict = asdict(config)
    
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def _override_from_env(config: ProjectConfig):
    """从环境变量覆盖配置"""
    # 数据配置
    if 'SPOOFING_DATA_ROOT' in os.environ:
        config.data.base_data_root = os.environ['SPOOFING_DATA_ROOT']
    
    # 模型配置
    if 'SPOOFING_USE_GPU' in os.environ:
        config.model.use_gpu = os.environ['SPOOFING_USE_GPU'].lower() == 'true'
    
    # 特征配置
    if 'SPOOFING_BACKEND' in os.environ:
        config.feature.backend = os.environ['SPOOFING_BACKEND']

# 默认配置实例
default_config = ProjectConfig()

# 便捷函数
def get_data_config() -> DataConfig:
    """获取数据配置"""
    return default_config.data

def get_model_config() -> ModelConfig:
    """获取模型配置"""
    return default_config.model

def get_feature_config() -> FeatureConfig:
    """获取特征配置"""
    return default_config.feature

def get_evaluation_config() -> EvaluationConfig:
    """获取评估配置"""
    return default_config.evaluation 