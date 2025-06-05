#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spoofing Detection Project - Main Entry Point
==============================================
统一的项目入口，提供简化的命令行接口
"""

import argparse
import sys
from pathlib import Path

# 添加core目录到路径
core_dir = Path(__file__).parent / "core"
sys.path.insert(0, str(core_dir))

def main():
    parser = argparse.ArgumentParser(
        description="Spoofing Detection Project - 虚假报单检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  
  # 完整的虚假报单检测流程（推荐）
  python main.py complete --data_root /path/to/data --tickers 000989.SZ 300233.SZ
  
  # 只运行训练（跳过数据处理）
  python main.py complete --data_root /path/to/data --skip_all
  
  # 参数优化
  python main.py optimize --data_root /path/to/data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # Complete Pipeline
    complete_parser = subparsers.add_parser('complete', help='完整的检测pipeline')
    complete_parser.add_argument('--data_root', required=True, help='数据根目录')
    complete_parser.add_argument('--tickers', nargs='*', help='股票列表')
    complete_parser.add_argument('--skip_all', action='store_true', help='跳过数据处理，直接训练')
    complete_parser.add_argument('--by_ticker', action='store_true', default=True, help='按股票分开训练')
    
    # Optimization Pipeline  
    opt_parser = subparsers.add_parser('optimize', help='参数优化pipeline')
    opt_parser.add_argument('--data_root', required=True, help='数据根目录')
    opt_parser.add_argument('--tickers', nargs='*', help='股票列表')
    
    args = parser.parse_args()
    
    if args.command == 'complete':
        from complete_spoofing_pipeline import CompleteSpoofingPipeline
        
        pipeline = CompleteSpoofingPipeline()
        
        # 构建参数
        kwargs = {
            'base_data_root': args.data_root,
            'train_regex': "202503|202504",
            'valid_regex': "202505",
            'tickers': args.tickers,
            'skip_merge': args.skip_all,
            'skip_etl': args.skip_all,
            'by_ticker': args.by_ticker
        }
        
        pipeline.run_complete_pipeline(**kwargs)
        
    elif args.command == 'optimize':
        from run_optimization_pipeline import main as opt_main
        
        # 设置优化参数
        sys.argv = [
            'run_optimization_pipeline.py',
            '--base_data_root', args.data_root,
            '--skip_merge', '--skip_etl'
        ]
        if args.tickers:
            sys.argv.extend(['--tickers'] + args.tickers)
        
        opt_main()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 