#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型保存辅助脚本
==============

将训练好的LightGBM模型和特征列表保存为适合manipulation_detection_heatmap.py使用的格式。

使用方法：
1. 训练完成后，调用save_model_for_analysis函数
2. 或者运行此脚本独立保存现有模型

示例：
python scripts/analysis/save_model_for_analysis.py \
  --model_path "path/to/trained_model.pkl" \
  --features_list "feature1,feature2,feature3" \
  --output_dir "results/saved_models"
"""

import argparse
import json
import os
import pickle
import joblib
from pathlib import Path
from typing import List, Union

import lightgbm as lgb
import pandas as pd


def save_model_for_analysis(
    model, 
    feature_names: List[str], 
    output_dir: str, 
    model_name: str = "spoofing_model"
):
    """
    保存模型和特征列表，供分析脚本使用
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        output_dir: 输出目录
        model_name: 模型名称前缀
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 保存模型和特征到: {output_path}")
    
    # 保存特征列表
    features_path = output_path / f"{model_name}_features.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✅ 特征列表已保存: {features_path}")
    
    # 保存模型
    if hasattr(model, 'booster_'):
        # LightGBM sklearn接口
        model_path = output_path / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': feature_names,
                'model_type': 'lightgbm_sklearn'
            }, f)
        print(f"✅ LightGBM模型已保存: {model_path}")
        
        # 另外保存为LightGBM原生格式
        lgb_path = output_path / f"{model_name}.txt"
        model.booster_.save_model(str(lgb_path))
        print(f"✅ LightGBM原生格式已保存: {lgb_path}")
        
    elif hasattr(model, 'save_model'):
        # LightGBM原生接口
        lgb_path = output_path / f"{model_name}.txt"
        model.save_model(str(lgb_path))
        
        # 保存模型和特征的组合
        model_path = output_path / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': feature_names,
                'model_type': 'lightgbm_native'
            }, f)
        print(f"✅ LightGBM模型已保存: {model_path}")
        
    else:
        # 其他模型（sklearn等）
        model_path = output_path / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': feature_names,
                'model_type': 'sklearn'
            }, f)
        print(f"✅ 模型已保存: {model_path}")
    
    # 保存使用说明
    readme_path = output_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"""# 保存的模型文件

## 文件说明
- `{model_name}.pkl`: 完整模型和特征信息（推荐用于分析脚本）
- `{model_name}_features.json`: 特征名称列表
- `{model_name}.txt`: LightGBM原生格式（如果适用）

## 使用方法

### 在manipulation_detection_heatmap.py中使用：

```bash
python scripts/analysis/manipulation_detection_heatmap.py \\
  --data_root "/path/to/data" \\
  --model_path "{output_path}/{model_name}.pkl" \\
  --output_dir "results/manipulation_analysis"
```

或者使用原生格式：

```bash
python scripts/analysis/manipulation_detection_heatmap.py \\
  --data_root "/path/to/data" \\
  --model_path "{output_path}/{model_name}.txt" \\
  --model_features_path "{output_path}/{model_name}_features.json" \\
  --output_dir "results/manipulation_analysis"
```

## 模型信息
- 特征数量: {len(feature_names)}
- 保存时间: {pd.Timestamp.now()}
""")
    
    print(f"✅ 说明文档已保存: {readme_path}")
    print(f"🎯 模型保存完成！可以在manipulation_detection_heatmap.py中使用 --model_path {model_path}")


def load_and_resave_model(model_path: str, output_dir: str, feature_names: List[str] = None):
    """
    加载现有模型并重新保存为分析脚本兼容格式
    
    Args:
        model_path: 现有模型路径
        output_dir: 输出目录
        feature_names: 特征名称列表（如果模型中没有保存）
    """
    print(f"📦 加载现有模型: {model_path}")
    
    try:
        # 尝试不同的加载方法
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                model = model_data.get('model')
                features = model_data.get('features', feature_names)
            else:
                model = model_data
                features = feature_names
                
        elif model_path.endswith('.joblib'):
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                model = model_data.get('model')
                features = model_data.get('features', feature_names)
            else:
                model = model_data
                features = feature_names
                
        elif model_path.endswith('.txt'):
            # LightGBM原生格式
            model = lgb.Booster(model_file=model_path)
            features = feature_names
            
        else:
            raise ValueError(f"不支持的模型格式: {model_path}")
        
        if features is None:
            raise ValueError("无法获取特征列表，请提供feature_names参数")
        
        # 重新保存
        model_name = Path(model_path).stem
        save_model_for_analysis(model, features, output_dir, model_name)
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="保存模型供分析脚本使用")
    parser.add_argument("--model_path", required=True, help="现有模型路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--features_list", help="特征名称（逗号分隔）")
    parser.add_argument("--features_file", help="特征名称文件路径")
    
    args = parser.parse_args()
    
    # 获取特征列表
    feature_names = None
    if args.features_list:
        feature_names = [f.strip() for f in args.features_list.split(',')]
    elif args.features_file and os.path.exists(args.features_file):
        if args.features_file.endswith('.json'):
            with open(args.features_file, 'r') as f:
                feature_names = json.load(f)
        else:
            with open(args.features_file, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
    
    if feature_names:
        print(f"📋 加载了 {len(feature_names)} 个特征")
    else:
        print("⚠️ 未提供特征列表，将尝试从模型中提取")
    
    # 重新保存模型
    load_and_resave_model(args.model_path, args.output_dir, feature_names)


# 可以直接在训练脚本中调用的函数
def save_training_results(
    model, 
    feature_names: List[str], 
    results: dict, 
    output_base_dir: str = "results"
):
    """
    训练完成后保存所有结果
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        results: 训练结果字典
        output_base_dir: 输出基础目录
    """
    import datetime
    
    # 创建带时间戳的输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"spoofing_model_{timestamp}"
    
    # 保存模型
    save_model_for_analysis(model, feature_names, str(output_dir))
    
    # 保存训练结果
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 训练结果已保存: {results_path}")
    
    return str(output_dir)


if __name__ == "__main__":
    main()

"""
使用示例：

1. 重新保存现有模型：
python scripts/analysis/save_model_for_analysis.py \
  --model_path "results/lgb_model.pkl" \
  --output_dir "results/saved_for_analysis" \
  --features_list "feature1,feature2,feature3"

2. 在训练脚本中使用：
from scripts.analysis.save_model_for_analysis import save_training_results

# 训练完成后
output_dir = save_training_results(
    model=trained_model,
    feature_names=feature_columns,
    results=evaluation_metrics
)
print(f"模型已保存到: {output_dir}")
""" 