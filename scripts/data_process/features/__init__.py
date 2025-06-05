#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
features 模块
────────────────────────────────────────
特征生成模块 - 第二阶段特征计算
• 从委托事件流计算各类特征
• 支持批量处理多日期数据
• 模块化特征计算流水线
• 保留所有计算特征（不过滤）
"""

from .feature_generator import (
    FeatureGenerator,
    process_event_stream_directory
)

__all__ = [
    'FeatureGenerator',
    'process_event_stream_directory'
] 