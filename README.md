# Spoofing Detect

该项目用于在高频交易数据中检测股票市場中的 "Spoofing" 行为。仓库提供了完整的数据处理与模型训练脚本，包括原始数据合并、特征生成、标签制作以及模型训练。

## 环境准备

```bash
pip install -r requirements.txt
```

- 建议使用 Python 3.8 及以上版本。
- 依赖列表详见 `requirements.txt`。

## 数据目录结构

数据目录示例可参考 `data/base_data_note.md`，示意结构如下：

```
data/
├── base_data/           # 解压后的原始 CSV
├── event_stream/        # 合并后的委托事件流
├── features/            # 生成的特征文件
├── labels_enhanced/     # 生成的标签文件
└── ...
```

仓库提供 `data/exmaple` 目录作为样例。

## 快速开始

1. **合并原始数据**

```bash
python scripts/data_process/raw_data/merge_event_stream.py \
    --root "/path/to/data/base_data" \
    --tickers $(cat example_tickers.txt)
```

2. **生成特征**

```bash
python scripts/data_process/features/feature_generator.py \
    --input_dir "/path/to/data/event_stream" \
    --output_dir "/path/to/data/features" \
    --backend polars --extended
```

3. **生成标签**

```bash
python scripts/data_process/labels/label_generator.py \
    --input_dir "/path/to/data/event_stream" \
    --output_dir "/path/to/data/labels_enhanced" \
    --r1_ms 1000 --r2_ms 1000 --r2_mult 1.0 \
    --extended --backend pandas
```

4. **模型训练**

```bash
python scripts/train/train.py \
    --data_root "/path/to/data" \
    --train_regex "202503|202504" \
    --valid_regex "202505" \
    --sampling_method "none" \
    --use_ensemble \
    --eval_output_dir "results/train_results"
```

更多特征处理细节请参阅 `scripts/data_process/features/README.md`。
