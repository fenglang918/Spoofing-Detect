import pandas as pd
import numpy as np

# 读取标签文件
df = pd.read_parquet('data/labels_enhanced/labels_20250314.parquet')

print("=== 数据泄露检查 ===")
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print()

print("=== 目标变量分布 ===")
print(f"y_label分布: {df['y_label'].value_counts().to_dict()}")
print()

print("=== 可疑特征检查 ===")
# 检查flag列
if 'flag_R1' in df.columns:
    same_as_target = (df['y_label'] == df['flag_R1']).all()
    print(f"flag_R1与y_label完全相同: {same_as_target}")
    print(f"flag_R1分布: {df['flag_R1'].value_counts().to_dict()}")

if 'flag_R2' in df.columns:
    corr_r2 = df['y_label'].corr(df['flag_R2'].astype(int))
    print(f"flag_R2与y_label相关性: {corr_r2:.4f}")
    print(f"flag_R2分布: {df['flag_R2'].value_counts().to_dict()}")

print()
print("=== Enhanced标签相关性 ===")
for col in ['enhanced_spoofing_liberal', 'enhanced_spoofing_moderate', 'enhanced_spoofing_strict']:
    if col in df.columns:
        corr = df['y_label'].corr(df[col])
        print(f"{col}与y_label相关性: {corr:.4f}")
        print(f"{col}分布: {df[col].value_counts().to_dict()}")
        print() 