#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
labels 模块
────────────────────────────────────────
标签生成模块 - 第二阶段标签计算
• 从委托事件流生成欺骗标签
• 支持多种标签规则（R1, R2, 扩展规则）
• 批量处理多日期数据
• 模块化标签计算流水线
"""

from .label_generator import (
    LabelGenerator,
    process_event_stream_directory
)

__all__ = [
    'LabelGenerator',
    'process_event_stream_directory'
] 