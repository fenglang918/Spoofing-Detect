# 🕵️ Spoofing Detection Project

端到端的虚假报单检测系统，基于机器学习的高频交易异常行为识别。

## 📁 项目结构

```
📂 Spoofing Detection/
├── 📂 core/                    # 核心代码
│   ├── complete_spoofing_pipeline.py   # 完整检测pipeline
│   └── run_optimization_pipeline.py    # 参数优化pipeline
├── 📂 scripts/                 # 数据处理脚本
│   ├── data_process/          # 数据预处理
│   ├── analysis/              # 分析工具
│   └── evaluation/            # 评估工具
├── 📂 results/                 # 结果输出
│   └── archive/               # 历史结果
├── 📂 docs/                    # 文档
├── 📂 data/                    # 数据目录
└── main.py                     # 🎯 主入口文件
```

## 🚀 快速开始

### 安装依赖
```bash
pip install pandas numpy scikit-learn lightgbm polars
```

### 运行检测系统

#### 1. 完整流程（推荐）
```bash
# 从原始数据开始，完整的检测流程
python main.py complete \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data/base_data" \
  --tickers 000989.SZ 300233.SZ
```

#### 2. 快速训练
```bash
# 跳过数据处理，直接训练模型
python main.py complete \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data/base_data" \
  --skip_all
```

#### 3. 参数优化
```bash
# 运行参数优化实验
python main.py optimize \
  --data_root "/obs/users/fenglang/general/Spoofing Detect/data/base_data"
```

## 🎯 核心特性

### 🏷️ Extended Labels Strategy
- **5种虚假报单模式**：快速撤单冲击、价格操纵、虚假流动性、分层撤单、活跃时段异常
- **多层次标签**：Liberal（任意模式）/ Moderate（2+模式）/ Strict（3+模式）
- **Extended Liberal**: 主要检测策略，提供最全面的虚假报单识别

### 🧠 智能训练架构
- **按股票分开训练**：针对每只股票的特点优化模型
- **特征无泄露**：严格控制时间顺序，避免未来信息泄露
- **自动平衡**：处理极度不平衡的标签分布

### 📊 全面评估体系
- **PR-AUC**: 主要评估指标，适合不平衡数据
- **Precision@K**: 实用的Top-K精度评估
- **分股票性能分析**: 识别高性能和低性能股票

## 📈 结果解读

### 输出文件
- `extended_labels_performance.csv`: 各标签策略性能对比
- `by_ticker_results.csv`: 分股票详细结果
- `ticker_averages.csv`: 股票平均性能统计
- `extended_labels_analysis.json`: 标签分布分析

### 关键指标
- **Extended Liberal PR-AUC**: 主要展示结果
- **Precision@0.1%**: 实际应用中的精度
- **股票性能差异**: 识别适合检测的股票

## 🛠️ 高级用法

### 直接使用核心模块
```python
from core.complete_spoofing_pipeline import CompleteSpoofingPipeline

pipeline = CompleteSpoofingPipeline()
results = pipeline.run_complete_pipeline(
    base_data_root="/path/to/data",
    train_regex="202503|202504",
    valid_regex="202505",
    by_ticker=True
)
```

### 自定义虚假报单模式
参考 `core/complete_spoofing_pipeline.py` 中的模式定义，可以添加新的检测规则。

## 📚 系统架构

1. **数据合并** (`scripts/data_process/merge_order_trade.py`)
2. **特征工程** (`scripts/data_process/run_etl_from_event.py`)
3. **扩展标签生成** (自动生成5种虚假报单模式)
4. **模型训练** (LightGBM + 按股票分开训练)
5. **综合评估** (多维度性能分析)

## 🔧 配置选项

- `--data_root`: 数据根目录
- `--tickers`: 指定股票列表
- `--skip_all`: 跳过数据处理步骤
- `--by_ticker`: 启用按股票分开训练（默认开启）

## 📝 更新日志

- **v2.0**: 引入Extended Labels和按股票训练
- **v1.0**: 基础虚假报单检测系统

---

🎯 **推荐使用**: `python main.py complete --data_root /path/to/data --skip_all` 进行快速训练和评估。
