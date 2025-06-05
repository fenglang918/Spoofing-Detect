# 新数据处理架构说明

## 🎯 架构概述

重构后的数据处理系统采用模块化设计，将复杂的ETL流程分解为四个独立的阶段，每个阶段专注于特定的功能，提高了代码的可维护性和可扩展性。

## 📁 目录结构

```
scripts/data_process/
├── raw_data/                   # 阶段1: 原始数据处理
│   ├── __init__.py
│   └── merge_event_stream.py   # 合并委托和成交数据
├── features/                   # 阶段2: 特征生成
│   ├── __init__.py
│   └── feature_generator.py    # 特征计算和生成
├── labels/                     # 阶段3: 标签生成
│   ├── __init__.py
│   └── label_generator.py      # 标签计算和生成
├── analysis/                   # 阶段4: 数据整合分析
│   ├── __init__.py
│   └── data_integration.py     # 数据整合和质量分析
├── complete_pipeline.py        # 完整流水线
└── README_NEW_ARCHITECTURE.md  # 本文档
```

## 🔄 数据处理流程

### 阶段1: 原始数据处理 (`raw_data/`)

**功能**: 将原始的逐笔委托和逐笔成交数据合并为统一的委托事件流

**输入**: 
- `base_data/{YYYYMMDD}/{YYYYMMDD}/{TICKER}/逐笔委托.csv`
- `base_data/{YYYYMMDD}/{YYYYMMDD}/{TICKER}/逐笔成交.csv`
- `base_data/{YYYYMMDD}/{YYYYMMDD}/{TICKER}/行情.csv`

**输出**: 
- `event_stream/{YYYYMMDD}/委托事件流.csv`

**主要功能**:
- 自动编码检测和CSV读取
- 委托与成交事件合并
- 行情数据贴合（100ms网格）
- 基础特征计算（存活时间、价格偏离等）
- 数据质量检查和清洗

**使用示例**:
```bash
python scripts/data_process/raw_data/merge_event_stream.py \
    --root "/path/to/base_data" \
    --tickers 000001.SZ 000002.SZ \
    --dates 20230301 20230302
```

### 阶段2: 特征生成 (`features/`)

**功能**: 从委托事件流计算各类实时可观测特征

**输入**: 
- `event_stream/{YYYYMMDD}/委托事件流.csv`

**输出**: 
- `features/X_{YYYYMMDD}.parquet`

**主要功能**:
- 基础实时特征（盘口快照、价格特征、滚动窗口）
- 扩展实时特征（z_survival、价格动量、订单密度）
- 订单簿压力特征（book_imbalance、price_aggressiveness）
- 支持Polars/Pandas双后端
- **保留所有计算特征，不进行过滤**

**使用示例**:
```bash
python scripts/data_process/features/feature_generator.py \
    --input_dir "/path/to/event_stream" \
    --output_dir "/path/to/features" \
    --backend polars \
    --extended
```

### 阶段3: 标签生成 (`labels/`)

**功能**: 从委托事件流生成欺骗交易标签

**输入**: 
- `event_stream/{YYYYMMDD}/委托事件流.csv`

**输出**: 
- `labels/labels_{YYYYMMDD}.parquet`

**主要功能**:
- R1规则（快速撤单）
- R2规则（价格操纵）
- 扩展标签规则（可选）
- 支持参数自定义（时间阈值、价差倍数）
- **只保留主键和标签相关列**

**使用示例**:
```bash
python scripts/data_process/labels/label_generator.py \
    --input_dir "/path/to/event_stream" \
    --output_dir "/path/to/labels" \
    --r1_ms 50 --r2_ms 1000 --r2_mult 4.0 \
    --extended
```

### 阶段4: 数据整合分析 (`analysis/`)

**功能**: 整合特征和标签数据，进行全局质量分析和智能过滤

**输入**: 
- `features/X_*.parquet`
- `labels/labels_*.parquet`

**输出**: 
- `analysis/integrated_data.parquet` (原始整合数据)
- `analysis/filtered_data.parquet` (过滤后数据)
- `analysis/data_analysis.json` (分析报告)

**主要功能**:
- 特征和标签数据合并
- 全局数据质量分析
- 信息泄露特征检测
- 常数列、低方差、重复特征识别
- 智能过滤建议生成
- **在全量数据基础上统一分析和过滤**

**使用示例**:
```bash
python scripts/data_process/analysis/data_integration.py \
    --features_dir "/path/to/features" \
    --labels_dir "/path/to/labels" \
    --output_dir "/path/to/analysis" \
    --apply_filter
```

## 🚀 完整流水线

### 一键运行完整流程

```bash
python scripts/data_process/complete_pipeline.py \
    --base_data_dir "/path/to/base_data" \
    --output_root "/path/to/output" \
    --tickers 000001.SZ 000002.SZ \
    --backend polars \
    --extended_features \
    --r1_ms 50 --r2_ms 1000 --r2_mult 4.0
```

### 灵活的阶段控制

```bash
# 跳过某些阶段（比如已经完成的阶段）
python scripts/data_process/complete_pipeline.py \
    --base_data_dir "/path/to/base_data" \
    --output_root "/path/to/output" \
    --skip_stages "1,2"  # 跳过阶段1和2
```

## ✅ 新架构的优势

### 1. **模块化设计**
- 每个阶段功能单一、职责明确
- 便于调试、测试和维护
- 支持独立运行和组合使用

### 2. **灵活的处理策略**
- 阶段2、3保留所有计算结果，不过早过滤
- 阶段4在全量数据基础上统一分析过滤
- 避免了特征不一致的问题

### 3. **智能化质量控制**
- 多维度数据质量分析
- 自动检测信息泄露特征
- 基于全局统计的过滤建议

### 4. **高度可配置**
- 支持灵活的参数配置
- 可选择不同的处理后端
- 支持阶段跳过和独立运行

### 5. **生产就绪**
- 完善的错误处理和日志
- 进度条和状态反馈
- 详细的处理报告

## 📊 输出数据说明

### 最终训练数据

- **文件**: `analysis/filtered_data.parquet`
- **内容**: 经过质量分析和智能过滤的训练就绪数据
- **特征**: 只保留高质量、无信息泄露的特征
- **格式**: Parquet格式，便于高效读取

### 数据质量报告

- **文件**: `analysis/data_analysis.json`
- **内容**: 详细的数据质量分析结果
- **包括**: 基础统计、问题特征、信息泄露检测、过滤建议等

## 🔧 配置参数说明

### 通用参数
- `--tickers`: 股票代码筛选（可选）
- `--dates`: 指定处理日期（可选）
- `--backend`: 计算后端（polars/pandas）

### 特征相关
- `--extended_features`: 是否计算扩展特征

### 标签相关
- `--r1_ms`: R1规则时间阈值（默认50ms）
- `--r2_ms`: R2规则时间阈值（默认1000ms）
- `--r2_mult`: R2规则价差倍数（默认4.0）
- `--extended_labels`: 是否使用扩展标签规则

### 控制参数
- `--skip_stages`: 跳过指定阶段（如"1,2"）

## 🎯 最佳实践

### 1. 开发阶段
```bash
# 先用小数据集测试
python scripts/data_process/complete_pipeline.py \
    --base_data_dir "/path/to/small_data" \
    --output_root "/path/to/test_output" \
    --dates 20230301 \
    --tickers 000001.SZ
```

### 2. 生产环境
```bash
# 完整数据处理
python scripts/data_process/complete_pipeline.py \
    --base_data_dir "/path/to/full_data" \
    --output_root "/path/to/production_output" \
    --backend polars \
    --extended_features \
    --extended_labels
```

### 3. 增量处理
```bash
# 只处理新增日期
python scripts/data_process/complete_pipeline.py \
    --base_data_dir "/path/to/data" \
    --output_root "/path/to/output" \
    --dates 20230315 20230316 \
    --skip_stages "4"  # 跳过分析，后续统一分析
```

## 🔍 故障排除

### 常见问题

1. **编码错误**: 使用自动编码检测，支持utf-8-sig/gbk/latin1
2. **内存不足**: 使用Polars后端，支持Lazy计算
3. **文件缺失**: 详细的错误提示和跳过机制
4. **特征不一致**: 新架构完全避免了这个问题

### 调试建议

1. 使用小数据集先测试
2. 逐阶段运行，检查中间结果
3. 查看详细的日志输出
4. 检查生成的分析报告

---

**注意**: 这个新架构完全替代了原有的ETL流程，提供了更好的模块化、可维护性和数据质量控制。建议在生产环境中逐步迁移到新架构。 