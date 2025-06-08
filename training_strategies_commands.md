# 🚀 Spoofing Detection 训练策略命令参考

本文档记录了在Spoofing Detection项目中测试的各种训练策略及其对应的命令行参数。

## 📋 基础命令结构

```bash
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  [其他策略参数]
```

---

## 🎯 1. 数据平衡策略

### 1.1 无采样（推荐）⭐
```bash
# 使用全量数据，保持原始1:171不平衡比例
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none"
```
**结果**: PR-AUC=0.035, Precision@0.1%=13.37%, 训练时间=63s

### 1.2 下采样（1:10比例）
```bash
# 将负样本下采样到1:10比例
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "undersample"
```
**结果**: PR-AUC=0.031, Precision@0.1%=8.74%, 训练时间=60s

### 1.3 分层下采样
```bash
# 按股票分别进行下采样
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "stratified_undersample"
```

---

## ⚖️ 2. 类别权重策略

### 2.1 理论权重（N_neg/N_pos）
```bash
# 使用理论计算的权重 ≈ 171
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_class_weight
```
**结果**: PR-AUC=0.019, Precision@0.1%=1.85%, 训练时间=66s

### 2.2 保守权重（0.5倍理论值）
```bash
# 使用保守权重，防止过拟合
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_class_weight \
  --class_weight_ratio 85
```
**结果**: PR-AUC=0.014, Precision@0.1%=0.99%, 训练时间=84s

### 2.3 轻量权重
```bash
# 使用较小的权重值
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_class_weight \
  --class_weight_ratio 20
```
**结果**: PR-AUC=0.018, Precision@0.1%=0.33%, 训练时间=63s

### 2.4 自定义权重
```bash
# 指定任意权重值
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_class_weight \
  --class_weight_ratio [YOUR_WEIGHT]
```

---

## 🎯 3. Focal Loss策略

### 3.1 标准Focal Loss
```bash
# 使用标准Focal Loss参数
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0
```
**结果**: PR-AUC=0.014, Precision@0.1%=1.39%, 训练时间=174s

### 3.2 调整Focal Loss参数
```bash
# 自定义alpha和gamma参数
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_focal_loss \
  --focal_alpha 0.5 \
  --focal_gamma 1.5
```

---

## 🔗 4. 集成学习策略

### 4.1 多模型集成（推荐）⭐
```bash
# LightGBM + XGBoost + RandomForest集成
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_ensemble
```
**结果**: PR-AUC=0.037, ROC-AUC=0.796, Precision@0.1%=11.58%, 训练时间=130s

### 4.2 集成+超参数优化
```bash
# 集成学习结合超参数调优
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_ensemble \
  --optimize_params \
  --n_trials 50
```

---

## 🔧 5. 超参数优化

### 5.1 单模型超参数优化
```bash
# 仅对LightGBM进行超参数优化
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --optimize_params \
  --n_trials 100
```

### 5.2 快速参数搜索
```bash
# 减少试验次数，快速搜索
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --optimize_params \
  --n_trials 20
```

---

## 🏷️ 6. 增强标签策略

### 6.1 使用增强标签
```bash
# 使用enhanced_spoofing_liberal标签
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_enhanced_labels \
  --label_type "enhanced_spoofing_liberal"
```

### 6.2 保守增强标签
```bash
# 使用enhanced_spoofing_strict标签
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_enhanced_labels \
  --label_type "enhanced_spoofing_strict"
```

---

## 🔄 7. 组合策略

### 7.1 最优组合（生产推荐）
```bash
# 集成学习 + 无采样 + 原始标签
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_ensemble
```

### 7.2 高精度单模型
```bash
# 单模型 + 无采样 + 超参数优化
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --optimize_params \
  --n_trials 50
```

### 7.3 快速原型
```bash
# 下采样 + 单模型，快速验证
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "undersample"
```

---

## 📊 8. 性能对比表

| 策略 | 命令关键参数 | PR-AUC | P@0.1% | 训练时间 | 推荐指数 |
|------|-------------|--------|--------|----------|----------|
| 🥇 集成学习 | `--use_ensemble --sampling_method none` | 0.037 | 11.58% | 130s | ⭐⭐⭐⭐⭐ |
| 🥈 无采样单模型 | `--sampling_method none` | 0.035 | 13.37% | 63s | ⭐⭐⭐⭐ |
| 🥉 1:10下采样 | `--sampling_method undersample` | 0.031 | 8.74% | 60s | ⭐⭐⭐ |
| 类别权重 | `--use_class_weight` | 0.019 | 1.85% | 66s | ⭐ |
| Focal Loss | `--use_focal_loss` | 0.014 | 1.39% | 174s | ⭐ |

---

## 💡 使用建议

### 🎯 生产环境
- **主力**: 集成学习策略
- **备选**: 无采样单模型（更快的预测速度）

### 🧪 实验阶段  
- **快速验证**: 下采样策略
- **参数调优**: 超参数优化策略

### 🚫 不推荐
- 类别权重策略（在此数据集上效果不佳）
- Focal Loss策略（计算复杂，效果不佳）

---

## 🔧 命令参数详解

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--data_root` | 数据根目录 | 路径 | 必需 |
| `--train_regex` | 训练数据日期正则 | 正则表达式 | "202503\|202504" |
| `--valid_regex` | 验证数据日期正则 | 正则表达式 | "202505" |
| `--sampling_method` | 采样方法 | none/undersample/stratified_undersample | undersample |
| `--use_ensemble` | 使用集成学习 | flag | False |
| `--use_class_weight` | 使用类别权重 | flag | False |
| `--class_weight_ratio` | 自定义权重比例 | 数值 | 自动计算 |
| `--use_focal_loss` | 使用Focal Loss | flag | False |
| `--focal_alpha` | Focal Loss alpha参数 | 0-1 | 0.25 |
| `--focal_gamma` | Focal Loss gamma参数 | >0 | 2.0 |
| `--optimize_params` | 超参数优化 | flag | False |
| `--n_trials` | 优化试验次数 | 整数 | 50 |
| `--use_enhanced_labels` | 使用增强标签 | flag | False |
| `--label_type` | 增强标签类型 | liberal/strict | liberal |

---

## 🚀 快速开始

```bash
# 1. 最佳性能（推荐生产使用）
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/path/to/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_ensemble

# 2. 快速验证（开发测试）
python scripts/train/train_baseline_enhanced_fixed.py \
  --data_root "/path/to/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "undersample"
```

---

*最后更新: 2024年* | *项目: Spoofing Detection* | *状态: 生产就绪* 