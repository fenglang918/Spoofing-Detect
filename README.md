# Spoofing Detection System

一个基于机器学习的金融市场欺骗交易检测系统，专门用于识别和分析股票交易中的欺骗行为模式。

## 项目简介

本项目是一个端到端的欺骗交易检测系统，涵盖从原始数据处理、特征工程、标签生成到模型训练和预测可视化的完整流程。系统主要针对中国A股市场的高频交易数据进行分析。

### 主要功能

- **数据处理**: 处理原始交易流数据，合并和清理事件流
- **标签生成**: 基于交易模式和时间窗口自动生成欺骗行为标签
- **特征工程**: 提取多维度交易特征用于机器学习
- **模型训练**: 支持多种机器学习算法和集成学习
- **预测分析**: 提供模型预测结果的可视化分析

## 技术栈

- **数据处理**: Pandas, Polars
- **机器学习**: Scikit-learn, LightGBM, XGBoost
- **数据不平衡处理**: Imbalanced-learn
- **超参数优化**: Optuna
- **可视化**: Matplotlib, Seaborn
- **开发工具**: Jupyter, Rich

## 项目结构

```
Spoofing Detect/
├── scripts/                    # 核心脚本目录
│   ├── data_process/           # 数据处理模块
│   │   ├── raw_data/          # 原始数据处理
│   │   ├── labels/            # 标签生成
│   │   ├── features/          # 特征工程
│   │   └── analysis/          # 数据分析
│   ├── train/                 # 模型训练
│   ├── analysis/              # 结果分析和可视化
│   └── utils/                 # 工具函数
├── data/                      # 数据目录
│   ├── base_data/            # 基础数据
│   ├── event_stream/         # 事件流数据
│   ├── labels_enhanced/      # 增强标签
│   └── features/             # 特征数据
├── results/                   # 结果输出目录
│   ├── trained_models/       # 训练好的模型
│   └── prediction_visualization/ # 预测可视化结果
├── latex/                     # LaTeX排版和报告
├── requirements.txt           # Python依赖
└── example_tickers.txt       # 示例股票代码
```

## 安装说明

### 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd "Spoofing Detect"
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

### GPU支持（可选）

如需GPU加速，请取消注释 `requirements.txt` 中的CUDA相关依赖：
```bash
pip install cudf>=23.0.0 cuml>=23.0.0
```

## 使用指南

### 完整流程示例

以下是一个完整的数据处理和模型训练流程示例：

#### 1. 数据预处理
合并原始事件流数据：
```bash
python scripts/data_process/raw_data/merge_event_stream.py \
    --root "/path/to/your/data/base_data" \
    --tickers 300233.SZ \
    --filter-hours
```

#### 2. 标签生成
生成增强标签用于训练：
```bash
python scripts/data_process/labels/label_generator.py \
    --input_dir "/path/to/your/data/event_stream" \
    --output_dir "/path/to/your/data/labels_enhanced" \
    --r1_ms 1000 --r2_ms 1000 --r2_mult 1.0 \
    --tickers 300233.SZ \
    --extended \
    --backend polars
```

#### 3. 特征工程
提取交易特征：
```bash
python scripts/data_process/features/feature_generator.py \
    --input_dir "/path/to/your/data/event_stream" \
    --output_dir "/path/to/your/data/features" \
    --tickers 300233.SZ \
    --backend polars \
    --extended
```

#### 4. 模型训练
训练欺骗检测模型：
```bash
python scripts/train/train.py \
    --data_root "/path/to/your/data" \
    --train_regex "202503|202504" \
    --valid_regex "202505" \
    --sampling_method "none" \
    --use_ensemble \
    --eval_output_dir "results/train_results"
```

#### 5. 预测可视化
生成预测结果可视化：
```bash
python scripts/analysis/model_prediction_visualization.py \
    --data_root "/path/to/your/data" \
    --model_path "results/trained_models/spoofing_model_Enhanced_none_Ensemble.pkl" \
    --valid_regex "202505" \
    --output_dir "results/prediction_visualization/202505_full_month" \
    --prob_threshold 0.01 \
    --top_k_percent 0.005 \
    --max_plots 50
```

### 参数说明

#### 数据相关参数
- `--tickers`: 股票代码列表，支持多个代码用空格分隔
- `--root` / `--input_dir` / `--output_dir`: 输入输出目录路径
- `--backend`: 数据处理后端，支持 `pandas` 或 `polars`

#### 标签生成参数
- `--r1_ms` / `--r2_ms`: 时间窗口参数（毫秒）
- `--r2_mult`: 价格变动倍数阈值
- `--extended`: 启用扩展特征

#### 训练参数
- `--train_regex` / `--valid_regex`: 训练和验证数据的时间范围正则表达式
- `--sampling_method`: 数据采样方法
- `--use_ensemble`: 启用集成学习

#### 可视化参数
- `--prob_threshold`: 概率阈值
- `--top_k_percent`: 显示前k%的预测结果
- `--max_plots`: 最大图表数量

## 数据格式

### 输入数据格式
- 原始数据应为事件流格式，包含时间戳、价格、数量等字段
- 支持的股票代码格式：`XXXXXX.SZ`（深交所）、`XXXXXX.SH`（上交所）

### 输出结果
- 训练好的模型保存在 `results/trained_models/` 目录
- 预测可视化结果保存在 `results/prediction_visualization/` 目录
- 支持多种格式的输出（pkl模型文件、图表、分析报告等）

## 性能优化

- 使用 `--backend polars` 可以获得更好的数据处理性能
- 启用 `--use_ensemble` 可以提高模型准确性但会增加训练时间
- 合理设置 `--max_plots` 参数以平衡分析深度和计算时间

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 [MIT许可证](LICENSE) - 详细信息请查看LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 GitHub Issue
- 发送邮件至项目维护者

## 更新日志

### v1.0.0
- 初始版本发布
- 完整的数据处理流程
- 集成机器学习模型
- 预测可视化功能