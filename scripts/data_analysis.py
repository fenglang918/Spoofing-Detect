import pandas as pd
import numpy as np
import os
import logging
import argparse
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载和处理类"""
    
    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # 验证数据目录是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
            
        # 定义数据文件映射
        self.data_files = {
            'market': '行情.csv',
            'order': '逐笔委托.csv',
            'transaction': '逐笔成交.csv'
        }
        
    def load_data(self, data_type: str, encoding: str = 'gbk') -> pd.DataFrame:
        """
        加载指定类型的数据
        
        Args:
            data_type: 数据类型 ('market', 'order', 'transaction')
            encoding: 文件编码
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        if data_type not in self.data_files:
            raise ValueError(f"不支持的数据类型: {data_type}")
            
        if data_type in self.data_cache:
            return self.data_cache[data_type]
            
        file_path = self.data_dir / self.data_files[data_type]
        try:
            logger.info(f"正在加载数据: {file_path}")
            df = pd.read_csv(file_path, encoding=encoding)
            self.data_cache[data_type] = df
            logger.info(f"成功加载数据: {data_type}, 形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {file_path}, 错误: {str(e)}")
            raise

    def print_data_info(self, data_type: str):
        """
        打印数据的详细信息
        
        Args:
            data_type: 数据类型
        """
        df = self.load_data(data_type)
        print(f"\n{'='*50}")
        print(f"{data_type} 数据信息:")
        print(f"{'='*50}")
        df.info()
        print(f"\n前5行数据预览:")
        print(df.head())
        print(f"\n{'='*50}\n")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据分析工具')
    parser.add_argument('--data-dir', type=str, default='data/exmaple',
                      help='数据目录路径 (默认: data/exmaple)')
    return parser.parse_args()

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 初始化数据加载器
        loader = DataLoader(args.data_dir)
        
        # 打印所有数据的信息
        for data_type in ['market', 'order', 'transaction']:
            loader.print_data_info(data_type)
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()

