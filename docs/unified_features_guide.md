# 统一特征计算模块使用指南

## 📖 概述

统一特征计算模块 (`unified_features.py`) 提供了一个标准化的特征计算接口，解决了以下问题：

- ✅ **统一接口**: 无论是单独运行还是ETL集成，都使用相同的特征计算逻辑
- ✅ **多后端支持**: 支持Polars（高性能）和Pandas（兼容性）
- ✅ **特征一致性**: 通过白名单机制确保所有日期文件特征一致
- ✅ **模块化设计**: 可扩展的特征计算流水线
- ✅ **生产就绪**: 自动移除信息泄露特征

## 🚀 快速开始

### 1. 作为独立脚本运行

```bash
# 计算单个文件的特征
python scripts/data_process/unified_features.py \
    --input "/path/to/委托事件流.csv" \
    --output "/path/to/features.parquet" \
    --tickers 000001.SZ 000002.SZ \
    --backend polars \
    --extended

# 查看帮助
python scripts/data_process/unified_features.py --help
```

### 2. 在Python代码中集成使用

```python
from scripts.data_process.unified_features import UnifiedFeatureCalculator

# 创建特征计算器
calculator = UnifiedFeatureCalculator(
    backend="polars",           # 或 "pandas"
    extended_features=True      # 是否计算扩展特征
)

# 计算特征
features_df = calculator.calculate_features(
    data=raw_event_data,
    tickers={'000001.SZ', '000002.SZ'},  # 可选：股票筛选
    apply_whitelist=True,                # 是否应用特征白名单
    show_progress=True                   # 是否显示进度条
)
```

### 3. 在ETL流水线中集成

ETL脚本已经自动集成了统一特征计算模块，无需额外配置：

```bash
python scripts/data_process/run_etl_from_event_refactored.py \
    --root "/path/to/event_stream" \
    --backend polars \
    --extended_labels
```

## 🔧 配置选项

### 后端选择

| 后端 | 优势 | 适用场景 |
|------|------|----------|
| `polars` | 高性能、内存效率高 | 大数据量、生产环境 |
| `pandas` | 兼容性好、调试方便 | 小数据量、开发测试 |

### 特征模式

| 模式 | 描述 | 特征数量 |
|------|------|----------|
| 基础特征 | 核心实时可观测特征 | ~25个 |
| 扩展特征 | 包含高级衍生特征 | ~35个 |

## 📊 特征类别

### 核心特征（始终包含）

```python
核心特征 = [
    # 主键
    '自然日', 'ticker', '交易所委托号',
    
    # 盘口快照
    'bid1', 'ask1', 'mid_price', 'spread', 'prev_close', 'bid_vol1', 'ask_vol1',
    
    # 订单静态特征
    'log_qty', 'is_buy',
    
    # 短期历史窗口
    'orders_100ms', 'orders_1s', 'cancels_100ms', 'cancels_1s', 'cancels_5s',
    'cancel_ratio_100ms', 'cancel_ratio_1s', 'trades_1s',
    
    # 时间周期特征
    'time_sin', 'time_cos', 'in_auction',
    
    # 价格相关
    'delta_mid', 'pct_spread', 'price_dev_prevclose_bps',
    
    # 衍生指标
    'book_imbalance', 'price_aggressiveness', 'cluster_score',
    
    # 事件标记
    'is_cancel_event'
]
```

### 扩展特征（可选）

```python
扩展特征 = [
    'z_survival',           # 异常生存时间
    'price_momentum_100ms', # 短期价格动量
    'spread_change',        # 价差变化
    'order_density',        # 订单密度
    'layering_score'        # 分层挂单评分
]
```

## 🛡️ 特征白名单机制

### 自动移除的特征（黑名单）

```python
黑名单特征 = [
    'is_cancel$',              # 原始撤单标志（信息泄露）
    'total_events',            # 总事件数（包含未来信息）
    'total_traded_qty',        # 总成交量（包含未来信息）
    'num_trades',              # 成交次数（包含未来信息）
    'num_cancels',             # 撤单次数（包含未来信息）
    'final_survival_time_ms',  # 最终存活时间（未来信息）
    'is_fully_filled',         # 是否完全成交（未来信息）
    'flag_R1', 'flag_R2'       # 中间标签变量
]
```

### 常数列检测

系统会自动检测并移除 `nunique() <= 1` 的常数列。

## 💡 使用示例

### 示例1：处理单个CSV文件

```python
from unified_features import UnifiedFeatureCalculator

# 创建计算器
calculator = UnifiedFeatureCalculator(backend="polars", extended_features=True)

# 处理文件
result = calculator.process_csv_file(
    csv_path="data/event_stream/20250101/委托事件流.csv",
    tickers={'000001.SZ'},
    output_path="data/features/X_20250101.parquet"
)

print(f"处理完成: {result.shape}")
```

### 示例2：批量处理

```python
from pathlib import Path
from unified_features import UnifiedFeatureCalculator

calculator = UnifiedFeatureCalculator(backend="polars", extended_features=True)

# 批量处理多个日期
event_root = Path("data/event_stream")
for date_dir in event_root.glob("2025*"):
    csv_file = date_dir / "委托事件流.csv"
    if csv_file.exists():
        calculator.process_csv_file(
            csv_path=csv_file,
            output_path=f"data/features/X_{date_dir.name}.parquet"
        )
```

### 示例3：自定义特征计算

```python
# 不应用白名单，保留所有特征
result = calculator.calculate_features(
    data=raw_data,
    apply_whitelist=False,
    show_progress=True
)

# 手动应用自定义筛选
filtered_result = result[my_custom_feature_list]
```

## 🔍 性能优化建议

### 1. 后端选择
- **大数据**: 使用Polars后端，内存使用更少，速度更快
- **小数据**: 使用Pandas后端，调试更方便

### 2. 股票筛选
```python
# ✅ 推荐：提前筛选股票
calculator.calculate_features(data, tickers={'000001.SZ', '000002.SZ'})

# ❌ 不推荐：处理所有股票后再筛选
result = calculator.calculate_features(data)
filtered = result[result['ticker'].isin(['000001.SZ', '000002.SZ'])]
```

### 3. 扩展特征
- 开发阶段：使用基础特征 (`extended_features=False`)
- 生产环境：使用扩展特征 (`extended_features=True`)

## 🚨 注意事项

### 1. 数据格式要求
输入数据必须包含以下列：
```python
必需列 = [
    'ticker', '委托_datetime', '事件_datetime', '委托价格', '委托数量',
    '方向_委托', '事件类型', '申买价1', '申卖价1', '前收盘',
    '申买量1', '申卖量1', '交易所委托号', '存活时间_ms'
]
```

### 2. 内存管理
- Polars后端会使用Lazy计算，内存效率更高
- 大数据量建议开启流式处理：`df.collect(streaming=True)`

### 3. 时间列格式
确保时间列为正确的datetime格式：
```python
df['委托_datetime'] = pd.to_datetime(df['委托_datetime'])
df['事件_datetime'] = pd.to_datetime(df['事件_datetime'])
```

## 🔧 故障排除

### 常见问题

1. **ImportError**: 确保安装了所需依赖
   ```bash
   pip install polars pandas rich
   ```

2. **KeyError**: 检查输入数据是否包含所有必需列

3. **MemoryError**: 对于大文件，使用Polars后端或增加系统内存

4. **特征不一致**: 确保应用了特征白名单 (`apply_whitelist=True`)

### 调试模式

```python
# 开启详细输出
calculator = UnifiedFeatureCalculator(backend="pandas", extended_features=False)
result = calculator.calculate_features(data, show_progress=True)

# 检查特征计算流水线
summary = calculator.get_feature_summary()
print(summary)
```

## 📈 集成到现有项目

### 1. 替换现有特征计算
```python
# 旧代码
df = calc_realtime_features(df)
df = calculate_enhanced_realtime_features(df)
df = apply_feature_whitelist(df)

# 新代码
calculator = UnifiedFeatureCalculator()
df = calculator.calculate_features(df)
```

### 2. 在ETL中集成
ETL脚本已自动集成，无需修改现有调用方式。

### 3. 添加自定义特征
扩展 `UnifiedFeatureCalculator` 类，添加自定义特征计算步骤：

```python
class CustomFeatureCalculator(UnifiedFeatureCalculator):
    def _build_feature_pipeline(self):
        pipeline = super()._build_feature_pipeline()
        pipeline.append({
            "name": "自定义特征",
            "function": "my_custom_features",
            "description": "我的自定义特征"
        })
        return pipeline
```

这样的设计确保了特征计算的一致性、可维护性和可扩展性！ 