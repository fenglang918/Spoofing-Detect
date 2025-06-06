# 🔍 Analysis Module

欺诈检测项目的分析模块，包含各种数据分析、模型评估和可视化工具。

## 📁 模块结构

### 🎯 热力图分析模块
**目录**: `heatmap_analysis/`

专门用于操纵行为检测和热力图生成的完整模块，包含：
- 主要分析脚本：`manipulation_detection_heatmap.py`
- 性能基准测试：`parallel_anomaly_benchmark.py`
- 快速启动脚本：`quick_heatmap_analysis.sh`
- 详细技术文档：`PARALLEL_OPTIMIZATION_README.md`

**快速开始**:
```bash
# 使用快速启动脚本
cd scripts/analysis/heatmap_analysis
./quick_heatmap_analysis.sh

# 或手动运行
python manipulation_detection_heatmap.py --data_root /path/to/data --model_path model.pkl
```

### 🔧 数据诊断工具

#### `diagnose_feat_label.py`
- 全面的特征-标签一致性检查
- 数据质量诊断和统计分析
- 异常检测和数据验证

```bash
python diagnose_feat_label.py --data_root /path/to/data
```

#### `analyze_data_leakage.py`
- 数据泄漏检测和分析
- 特征重要性时间稳定性检查
- 模型泛化能力评估

```bash
python analyze_data_leakage.py --data_root /path/to/data
```

### 📊 性能评估工具

#### `performance_comparison.py`
- 多模型性能对比分析
- 详细的评估指标计算
- 可视化性能报告生成

```bash
python performance_comparison.py --model_dir /path/to/models
```

#### `save_model_for_analysis.py`
- 模型保存和元数据管理
- 特征重要性分析
- 模型可解释性工具

```bash
python save_model_for_analysis.py --model_path model.pkl --output_dir results/
```

### 📈 交互式分析

#### `data_analysis.ipynb`
- Jupyter notebook格式的交互式数据分析
- 探索性数据分析(EDA)
- 可视化图表和统计报告

```bash
# 启动Jupyter notebook
jupyter notebook data_analysis.ipynb
```

## 🚀 快速使用指南

### 1. 数据质量检查
```bash
# 首先运行数据诊断
python diagnose_feat_label.py --data_root /path/to/data

# 检查数据泄漏
python analyze_data_leakage.py --data_root /path/to/data
```

### 2. 热力图分析
```bash
# 运行完整的操纵行为分析
cd heatmap_analysis/
./quick_heatmap_analysis.sh
```

### 3. 模型性能评估
```bash
# 模型性能对比
python performance_comparison.py --model_dir results/trained_models/

# 保存模型分析结果
python save_model_for_analysis.py --model_path best_model.pkl
```

## 📋 依赖安装

```bash
# 核心依赖
pip install pandas numpy scikit-learn lightgbm

# 可视化依赖
pip install matplotlib seaborn plotly

# 性能监控
pip install tqdm psutil

# Jupyter支持
pip install jupyter ipywidgets
```

## 🔧 配置说明

### 环境变量
```bash
export DATA_ROOT="/path/to/your/data"
export MODEL_ROOT="/path/to/your/models"
export RESULTS_ROOT="/path/to/your/results"
```

### 服务器环境优化
所有脚本都已针对无GUI的Linux服务器环境进行优化：
- 自动设置matplotlib后端为'Agg'
- 支持批处理和并行计算
- 详细的进度条和日志输出

## 📊 输出文件类型

### 热力图模块输出
- `*.png` - 各种热力图可视化
- `*.parquet` - 分析结果数据
- `*.txt` - 文本格式分析报告

### 诊断工具输出
- `*_diagnosis_report.txt` - 诊断报告
- `*_quality_metrics.json` - 数据质量指标
- `*_visualization.png` - 诊断可视化

### 性能评估输出
- `*_performance_report.json` - 性能指标
- `*_comparison_chart.png` - 对比图表
- `*_feature_importance.png` - 特征重要性图

## 🎯 使用场景

### 研发阶段
1. **数据验证**: 使用诊断工具确保数据质量
2. **特征工程**: 通过EDA发现有价值的特征
3. **模型调优**: 使用性能对比工具选择最佳模型

### 生产环境
1. **异常监控**: 使用热力图模块实时监控异常
2. **性能跟踪**: 定期运行性能评估
3. **数据漂移检测**: 使用数据泄漏检测工具

### 合规报告
1. **监管分析**: 生成操纵行为检测报告
2. **风险评估**: 使用热力图识别高风险时段和股票
3. **审计支持**: 提供详细的分析日志和报告

## 🆘 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size参数
   - 使用数据采样：`--sample_size 10000`

2. **CPU使用率低**
   - 增加并行线程数：`--max_workers 20`
   - 启用并发优化：不使用`--disable_parallel`

3. **GPU显示错误**
   - 服务器环境已自动设置为无GUI模式
   - 如仍有问题，设置：`export DISPLAY=""`

4. **模型加载失败**
   - 检查模型文件路径和格式
   - 确认特征列表文件存在
   - 查看详细错误日志

### 获取帮助
```bash
# 查看脚本帮助信息
python script_name.py --help

# 查看详细文档
cat README.md
cat heatmap_analysis/README.md
```

---

💡 **建议**: 建议按照"数据诊断 → 热力图分析 → 性能评估"的顺序使用各个模块，以获得最佳的分析效果。 