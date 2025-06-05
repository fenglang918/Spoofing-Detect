#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis 模块
────────────────────────────────────────
数据整合分析模块 - 第三阶段数据整合与评估
• 整合特征和标签数据
• 全局数据质量分析
• 特征质量评估和筛选
• 生成训练就绪数据
"""

from .data_integration import (
    DataIntegrator
)

__all__ = [
    'DataIntegrator'
] 