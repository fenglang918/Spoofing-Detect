#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_event_stream.py
────────────────────────────────────────
原始数据处理模块 - 第一阶段数据处理
• 合并逐笔委托和逐笔成交数据
• 贴合行情快照数据
• 数据清洗和标准化
• 生成统一的委托事件流
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import re
import sys
from typing import Set, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

console = Console()

# ────────────────────────── 工具函数 ──────────────────────────
def read_csv_auto(path: Path, **kw) -> pd.DataFrame:
    """自动尝试 utf-8-sig / gbk / latin1 读取 CSV"""
    for enc in ("utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kw)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"无法解码: {path}")

def prepare_quotes(quote_csv: Path) -> pd.DataFrame:
    """读取行情.csv → 100 ms 网格 → 返回含 quote_dt 的 DataFrame"""
    try:
        quotes = pd.read_csv(
            quote_csv, encoding="gbk",
            usecols=["自然日","时间","申买价1","申卖价1","申买量1","申卖量1","前收盘"]
        )
        quotes["quote_dt"] = pd.to_datetime(
            quotes["自然日"].astype(str) + quotes["时间"].astype(str).str.zfill(9),
            format="%Y%m%d%H%M%S%f"
        )
        quotes = (
            quotes.drop_duplicates("quote_dt")           # 去重
                  .set_index("quote_dt")
                  .resample("100ms")                     # 100 ms 网格
                  .ffill()                               # 前向填充
                  .reset_index()
        )
        return quotes
    except Exception as e:
        console.print(f"[yellow]警告: 行情数据处理失败 {quote_csv}: {e}[/yellow]")
        return pd.DataFrame()

def calculate_survival_time(merged_df: pd.DataFrame) -> pd.DataFrame:
    """计算订单存活时间"""
    merged_df["存活时间_ms"] = (
        merged_df["事件_datetime"] - merged_df["委托_datetime"]
    ).dt.total_seconds() * 1_000
    
    # 数据质量检查
    invalid_survival = merged_df["存活时间_ms"] < 0
    if invalid_survival.any():
        console.print(f"[yellow]警告: 发现 {invalid_survival.sum()} 条负存活时间记录[/yellow]")
        merged_df.loc[invalid_survival, "存活时间_ms"] = 0
    
    return merged_df

def add_order_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """添加基础订单特征"""
    # 委托金额
    merged_df["委托金额"] = pd.to_numeric(merged_df["委托价格"], errors='coerce') * pd.to_numeric(merged_df["委托数量"], errors='coerce')
    
    # 是否买单
    merged_df["is_buy"] = (merged_df["方向_委托"] == "B").astype(int)
    
    # 委托数量对数
    qty = pd.to_numeric(merged_df["委托数量"], errors='coerce')
    merged_df["log_qty"] = np.log1p(qty.fillna(0))
    
    # 价格相关特征
    委托价格 = pd.to_numeric(merged_df["委托价格"], errors='coerce')
    申买价1 = pd.to_numeric(merged_df["申买价1"], errors='coerce')
    申卖价1 = pd.to_numeric(merged_df["申卖价1"], errors='coerce')
    
    # 中间价
    merged_df["mid_price"] = (申买价1 + 申卖价1) / 2
    
    # 价差
    merged_df["spread"] = 申卖价1 - 申买价1
    merged_df["pct_spread"] = merged_df["spread"] / merged_df["mid_price"] * 10000  # bps
    
    # 价格偏离
    merged_df["price_vs_mid"] = 委托价格 - merged_df["mid_price"]
    merged_df["price_vs_mid_bps"] = merged_df["price_vs_mid"] / merged_df["mid_price"] * 10000
    
    return merged_df

def merge_one_stock(order_csv: Path, trade_csv: Path, ticker: str) -> pd.DataFrame:
    """单支股票：逐笔委托 + 逐笔成交 => 事件流（含存活时间）"""
    try:
        orders = read_csv_auto(order_csv, dtype=str)
        trades = read_csv_auto(trade_csv, dtype=str)
    except Exception as e:
        console.print(f"[red]错误: 读取文件失败 {ticker}: {e}[/red]")
        return pd.DataFrame()

    # 委托表预处理
    orders = orders.rename(columns={"委托代码": "方向_委托"})
    orders["交易所委托号"] = orders["交易所委托号"].str.lstrip("0")

    # 成交表拆买卖
    buy_cols  = ["自然日","时间","成交代码","成交价格","成交数量","叫买序号"]
    sell_cols = ["自然日","时间","成交代码","成交价格","成交数量","叫卖序号"]
    
    buy  = trades[trades["叫买序号"]  != "0"][buy_cols].copy() if "叫买序号" in trades.columns else pd.DataFrame()
    sell = trades[trades["叫卖序号"] != "0"][sell_cols].copy() if "叫卖序号" in trades.columns else pd.DataFrame()
    
    if not buy.empty:
        buy.rename(columns={"叫买序号" :"交易所委托号"}, inplace=True)
        buy["方向_事件"] = "B"
    
    if not sell.empty:
        sell.rename(columns={"叫卖序号":"交易所委托号"}, inplace=True)
        sell["方向_事件"] = "S"
    
    events = pd.concat([buy, sell], ignore_index=True)
    if events.empty:
        console.print(f"[yellow]警告: {ticker} 无有效成交事件[/yellow]")
        return pd.DataFrame()
    
    events["事件类型"] = np.where(events["成交代码"] == "C", "撤单", "成交")
    events["交易所委托号"] = events["交易所委托号"].str.lstrip("0")

    # 合并委托 + 成交
    merged = events.merge(
        orders,
        on="交易所委托号",
        how="left",
        suffixes=("_事件","_委托")
    )

    if merged.empty:
        console.print(f"[yellow]警告: {ticker} 合并后无数据[/yellow]")
        return pd.DataFrame()

    # 时间列处理
    fmt = "%Y%m%d%H%M%S%f"
    merged["委托_datetime"] = pd.to_datetime(
        merged["自然日_委托"].str.zfill(8) + merged["时间_委托"].str.zfill(9),
        errors="coerce", format=fmt)
    merged["事件_datetime"] = pd.to_datetime(
        merged["自然日_事件"].str.zfill(8) + merged["时间_事件"].str.zfill(9),
        errors="coerce", format=fmt)

    # 过滤无效时间
    merged = merged[merged["委托_datetime"].notna() & merged["事件_datetime"].notna()].copy()
    if merged.empty:
        console.print(f"[yellow]警告: {ticker} 时间解析后无有效数据[/yellow]")
        return pd.DataFrame()

    # 读取并重采样当日行情
    quote_csv = order_csv.with_name("行情.csv")
    quotes = prepare_quotes(quote_csv)
    
    if not quotes.empty:
        # 贴最近 ≤ t 的买一/卖一
        merged.sort_values("委托_datetime", inplace=True)
        merged = pd.merge_asof(
            merged, quotes,
            left_on="委托_datetime", right_on="quote_dt",
            direction="backward",
            tolerance=pd.Timedelta("100ms")
        )
    else:
        # 如果没有行情数据，添加空列避免后续处理错误
        for col in ["申买价1", "申卖价1", "申买量1", "申卖量1", "前收盘"]:
            merged[col] = np.nan

    # 计算存活时间
    merged = calculate_survival_time(merged)
    
    # 添加基础特征
    merged = add_order_features(merged)
    
    # 添加股票代码
    merged["ticker"] = ticker
    
    # 清理无用列
    merged = merged.loc[:, ~merged.columns.str.contains("^Unnamed")]
    
    return merged

def merge_one_day(date_dir: Path, watch_set: Set[str], out_root: Path) -> bool:
    """
    处理单日：date_dir = base_data/YYYYMMDD
    输出 event_stream/YYYYMMDD/委托事件流.csv
    """
    inner = date_dir / date_dir.name
    if not inner.exists():
        inner = date_dir
    
    dfs = []
    trading_date = date_dir.name
    
    # 获取股票目录列表
    stock_dirs = list(inner.iterdir()) if inner.exists() else []
    if watch_set:
        stock_dirs = [d for d in stock_dirs if d.name in watch_set]
    
    if not stock_dirs:
        console.print(f"[yellow]× {trading_date}: 无目标股票数据[/yellow]")
        return False
    
    console.print(f"[cyan]▶ 处理 {trading_date} ({len(stock_dirs)} 支股票)[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"处理股票", total=len(stock_dirs))
        
        for stk_dir in sorted(stock_dirs):
            progress.update(task, description=f"处理 {stk_dir.name}")
            
            o_csv = stk_dir / "逐笔委托.csv"
            t_csv = stk_dir / "逐笔成交.csv"
            
            if not (o_csv.exists() and t_csv.exists()):
                console.print(f"  [yellow]跳过 {stk_dir.name}: 缺少文件[/yellow]")
                progress.advance(task)
                continue
            
            df = merge_one_stock(o_csv, t_csv, stk_dir.name)
            if not df.empty:
                # 统一的交易日期字段
                df["自然日"] = trading_date
                
                # 数据质量检查
                if "自然日_委托" in df.columns:
                    inconsistent_dates = df[df["自然日_委托"] != trading_date]
                    if not inconsistent_dates.empty:
                        console.print(f"  [yellow]⚠️ {stk_dir.name}: {len(inconsistent_dates)} 条日期不一致记录[/yellow]")
                
                dfs.append(df)
            
            progress.advance(task)

    if not dfs:
        console.print(f"[yellow]× {trading_date}: 无有效数据[/yellow]")
        return False

    # 输出目录
    out_dir = out_root / trading_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "委托事件流.csv"
    
    # 合并所有股票数据
    final_df = pd.concat(dfs, ignore_index=True)
    
    # 重新排列列顺序，将主键字段放在前面
    key_cols = ["自然日", "ticker", "交易所委托号"]
    other_cols = [col for col in final_df.columns if col not in key_cols]
    final_df = final_df[key_cols + other_cols]
    
    # 最终数据质量检查
    duplicated = final_df.duplicated(subset=key_cols)
    if duplicated.any():
        console.print(f"  [yellow]⚠️ 发现 {duplicated.sum()} 条重复记录，已去重[/yellow]")
        final_df = final_df[~duplicated]
    
    # 保存
    final_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    
    console.print(f"[green]✓ {trading_date}: 已保存 → {out_file.name}[/green]")
    console.print(f"  📊 股票: {final_df['ticker'].nunique()}, 订单: {final_df['交易所委托号'].nunique():,}, 总行数: {len(final_df):,}")
    
    return True

def validate_data_structure(base_dir: Path) -> bool:
    """验证数据目录结构"""
    if not base_dir.exists():
        console.print(f"[red]错误: 根目录不存在 {base_dir}[/red]")
        return False
    
    date_dirs = [p for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)]
    if not date_dirs:
        console.print(f"[red]错误: 未找到任何日期目录 (YYYYMMDD) in {base_dir}[/red]")
        return False
    
    console.print(f"[green]✓ 发现 {len(date_dirs)} 个日期目录[/green]")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="原始数据处理模块 - 合并委托事件流")
    parser.add_argument("--root", required=True, help="base_data 根目录")
    parser.add_argument("--tickers", nargs="*", default=None, help="股票白名单，留空=全部")
    parser.add_argument("--output", help="输出根目录，默认为 {root}/../event_stream")
    parser.add_argument("--dates", nargs="*", help="指定处理日期，格式 YYYYMMDD")
    
    args = parser.parse_args()

    base = Path(args.root)
    if not validate_data_structure(base):
        sys.exit(1)

    watch_set = set(args.tickers) if args.tickers else set()
    out_root = Path(args.output) if args.output else base.parent / "event_stream"
    out_root.mkdir(exist_ok=True)

    console.print(f"\n[bold green]🚀 原始数据处理模块[/bold green]")
    console.print(f"[dim]输入目录: {base}[/dim]")
    console.print(f"[dim]输出目录: {out_root}[/dim]")
    console.print(f"[dim]股票筛选: {list(watch_set) if watch_set else '全部'}[/dim]")

    # 获取日期目录
    all_date_dirs = [p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)]
    
    if args.dates:
        # 筛选指定日期
        target_dates = set(args.dates)
        date_dirs = [d for d in all_date_dirs if d.name in target_dates]
        missing_dates = target_dates - {d.name for d in date_dirs}
        if missing_dates:
            console.print(f"[yellow]警告: 未找到日期目录 {missing_dates}[/yellow]")
    else:
        date_dirs = all_date_dirs

    if not date_dirs:
        console.print(f"[red]错误: 无可处理的日期目录[/red]")
        sys.exit(1)

    console.print(f"[dim]处理日期: {len(date_dirs)} 个[/dim]\n")

    # 处理数据
    successful_days = 0
    for d in sorted(date_dirs):
        if merge_one_day(d, watch_set, out_root):
            successful_days += 1

    # 总结
    console.print(f"\n[bold cyan]📊 处理完成[/bold cyan]")
    console.print(f"成功处理: {successful_days}/{len(date_dirs)} 天")
    console.print(f"输出目录: {out_root}")
    
    if successful_days == len(date_dirs):
        console.print(f"[bold green]✅ 所有数据处理成功！[/bold green]")
    else:
        console.print(f"[bold yellow]⚠️ 部分数据处理失败，请检查错误信息[/bold yellow]")

if __name__ == "__main__":
    main()

"""
使用示例:

# 处理所有日期、指定股票
python scripts/data_process/raw_data/merge_event_stream.py \
    --root "/home/ma-user/code/fenglang/Spoofing Detect/data/base_data" \
    --tickers 000989.SZ 300233.SZ

# 处理指定日期
python scripts/data_process/raw_data/merge_event_stream.py \
    --root "/path/to/base_data" \
    --dates 20230301 20230302 \
    --tickers 000001.SZ

# 自定义输出目录
python scripts/data_process/raw_data/merge_event_stream.py \
    --root "/path/to/base_data" \
    --output "/path/to/custom_output"
""" 