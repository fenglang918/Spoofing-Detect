# 特征处理模块

这个模块包含了从原始委托事件流生成特征和分析特征质量的完整流水线。

## 📁 模块结构

```
scripts/data_process/features/
├── __init__.py                 # 模块初始化
├── feature_generator.py        # 特征生成器（主要模块）
├── feature_analyzer.py         # 特征分析器
├── quick_view.py               # 分析结果查看器
└── README.md                   # 本文档
```

## 🚀 使用流程

### 1. 特征生成

从委托事件流生成机器学习特征：

```bash
# 处理所有日期的数据
python scripts/data_process/features/feature_generator.py \
    --input_dir "/path/to/event_stream" \
    --output_dir "/path/to/features" \
    --backend polars \
    --extended

# 处理指定日期和股票
python scripts/data_process/features/feature_generator.py \
    --input_dir "/path/to/event_stream" \
    --output_dir "/path/to/features" \
    --tickers 000001.SZ 000002.SZ \
    --dates 20230301 20230302 \
    --backend polars
```

**参数说明**：
- `--input_dir`: 委托事件流数据目录（event_stream）
- `--output_dir`: 特征输出目录
- `--backend`: 计算后端（polars 或 pandas，推荐 polars）
- `--extended`: 是否计算扩展特征
- `--tickers`: 指定处理的股票代码
- `--dates`: 指定处理的日期（格式：YYYYMMDD）

**输出**：
- 生成 `X_YYYYMMDD.parquet` 格式的特征文件
- 每个文件包含约60个特征
- 文件格式：Parquet（高效压缩，快速读取）

### 2. 特征分析

分析生成的特征数据质量：

```bash
# 基础分析
python scripts/data_process/features/feature_analyzer.py \
    --features_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/features" \
    --sample_files 10

# 自定义输出目录
python scripts/data_process/features/feature_analyzer.py \
    --features_dir "/path/to/features" \
    --output_dir "/path/to/analysis_output" \
    --sample_files 5

# 快速分析（跳过可视化）
python scripts/data_process/features/feature_analyzer.py \
    --features_dir "/path/to/features" \
    --no_viz
```

**参数说明**：
- `--features_dir`: 特征文件目录
- `--output_dir`: 分析结果输出目录（默认为 features_dir/analysis）
- `--sample_files`: 用于分析的样本文件数量（默认5个）
- `--no_viz`: 跳过可视化生成

**输出**：
- `feature_analysis_report.json`: 详细分析报告
- `visualizations/`: 可视化图表目录
  - `feature_distributions.png`: 特征分布图
  - `correlation_heatmap.png`: 相关性热力图
  - `missing_values.png`: 缺失值分析图

### 3. 结果查看

友好地查看分析结果：

```bash
# 查看默认位置的分析报告
python scripts/data_process/features/quick_view.py

# 查看指定报告
python scripts/data_process/features/quick_view.py /path/to/report.json
```

## 📊 特征类型

生成的特征包括以下几个类别：

### 基础特征
- 委托信息：价格、数量、方向、类型
- 成交信息：成交价格、数量、类型
- 时间特征：委托时间、成交时间、存活时间

### 市场特征
- 盘口快照：买一/卖一价格和数量
- 价格特征：中间价、价差、价格偏离
- 市场压力：订单簿不平衡、价格激进性

### 技术特征
- 滚动统计：均值、标准差、最值
- 时间窗口特征：多时间尺度统计
- 相对特征：价格相对位置、数量比例

### 高级特征
- Z-score标准化特征
- 价格动量特征
- 订单密度特征
- 流动性相关特征

## 🔧 技术架构

### 计算后端
- **Polars** (推荐)：高性能数据处理，内存效率高
- **Pandas**：传统数据处理，兼容性好

### 特征流水线
```python
数据加载 → 预处理 → 基础特征 → 扩展特征 → 订单簿特征 → 输出保存
```

### 数据质量保证
- 自动数据类型处理
- 时间序列排序
- 异常值检测
- 缺失值处理

## 📈 分析报告内容

### 数据质量分析
- 缺失值统计：识别数据完整性问题
- 异常值检测：发现潜在的数据质量问题
- 特征类型分布：了解数据结构

### 特征相关性分析
- 相关系数矩阵：识别冗余特征
- 高相关特征对：特征选择参考
- 相关性可视化：直观理解特征关系

### 时间序列分析
- 日度统计：数据量和质量的时间趋势
- 特征稳定性：特征在时间维度的变化
- 数据一致性：跨时间的数据质量

## ⚠️ 注意事项

### 性能优化
- 大数据量时建议使用 Polars 后端
- 适当调整采样文件数量
- 可以分批处理避免内存不足

### 数据质量
- 定期检查特征分析报告
- 关注高异常值特征（>10%）
- 监控高相关性特征对（|r|>0.8）

### 存储管理
- Parquet格式高效压缩
- 定期清理临时文件
- 合理规划存储空间

## 🐛 问题排查

### 常见错误

1. **数据类型解析失败**
   ```
   could not parse `U` as dtype `i64`
   ```
   - 解决：已在代码中处理，使用 schema_overrides

2. **内存不足**
   ```
   MemoryError: Unable to allocate array
   ```
   - 解决：减少 sample_files 数量或使用 Polars 后端

3. **排序错误**
   ```
   argument in operation 'rolling' is not sorted
   ```
   - 解决：已在代码中自动排序

### 性能调优
- 使用 `--backend polars` 获得最佳性能
- 调整 `--sample_files` 参数平衡分析精度和速度
- 在SSD上运行以获得更好的I/O性能

## 📝 示例脚本

完整的处理流程示例：

```bash
#!/bin/bash

# 1. 生成特征
echo "🚀 开始特征生成..."
python scripts/data_process/features/feature_generator.py \
    --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/features" \
    --backend polars \
    --extended

# 2. 分析特征质量
echo "📊 开始特征分析..."
python scripts/data_process/features/feature_analyzer.py \
    --features_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/features" \
    --sample_files 10

# 3. 查看分析结果
echo "👁️ 查看分析结果..."
python scripts/data_process/features/quick_view.py

echo "✅ 特征处理完成！"
```

## 🔗 相关模块

- `scripts/data_process/raw_data/merge_event_stream.py`: 原始数据合并
- `scripts/data_process/feature_engineering/`: 特征工程函数库
- `scripts/modeling/`: 模型训练相关模块

---

📞 **技术支持**: 如果遇到问题，请检查日志输出或查看相关源代码注释。 