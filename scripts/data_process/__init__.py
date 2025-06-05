#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_process package
────────────────────────────────────────
欺诈检测ETL流水线模块化包

模块组成:
- utils.py: 工具函数和配置
- feature_engineering.py: 特征工程模块
- labeling.py: 标签生成模块
- polars_engine.py: Polars高性能处理引擎
- pandas_engine.py: Pandas处理引擎
- run_etl_from_event_refactored.py: 重构后的主流水线
"""

__version__ = "2.0.0"
__author__ = "Spoofing Detection Team"

# 导出主要接口
from .run_etl_from_event_refactored import main as run_etl_main
from .utils import apply_feature_whitelist
from .feature_engineering import (
    calc_realtime_features, 
    calc_realtime_features_polars,
    calculate_enhanced_realtime_features
)
from .labeling import (
    detect_layering_pattern,
    improved_spoofing_rules,
    improved_spoofing_rules_polars
)

__all__ = [
    'run_etl_main',
    'apply_feature_whitelist',
    'calc_realtime_features',
    'calc_realtime_features_polars', 
    'calculate_enhanced_realtime_features',
    'detect_layering_pattern',
    'improved_spoofing_rules',
    'improved_spoofing_rules_polars'
] 