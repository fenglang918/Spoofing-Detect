#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_order_trade.py
────────────────────
功能：
1. 将 <逐笔委托.csv> 与 <逐笔成交.csv> 合并生成 "委托事件流.csv"
2. 先尝试 utf-8-sig → gbk → latin1 自动识别编码
3. 可指定 watch_list；留空=全部股票
4. 批量处理 base_data/下所有日期

目录预期：
base_data/
  20250303/20250303/000989.SZ/逐笔委托.csv
                                 逐笔成交.csv
  20250304/20250304/300233.SZ/...
  ...

运行示例（全部日期、两只股票）:
    python merge_order_trade.py \
        --root "/obs/users/fenglang/general/Spoofing Detect/data/base_data" \
        --tickers 000989.SZ 300233.SZ
"""

import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
import argparse, re, sys

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
              .resample("100ms")                     # —— ① 关键：100 ms 网格
              .ffill()                               # —— ② 前向填充
              .reset_index()
    )
    return quotes

def merge_one_stock(order_csv: Path, trade_csv: Path) -> pd.DataFrame:
    """单支股票：逐笔委托 + 逐笔成交 => 事件流（含存活时间）"""
    orders = read_csv_auto(order_csv, dtype=str)
    trades = read_csv_auto(trade_csv, dtype=str)

    # 委托表预处理
    orders = orders.rename(columns={"委托代码": "方向_委托"})
    orders["交易所委托号"] = orders["交易所委托号"].str.lstrip("0")

    # 成交表拆买卖
    buy_cols  = ["自然日","时间","成交代码","成交价格","成交数量","叫买序号"]
    sell_cols = ["自然日","时间","成交代码","成交价格","成交数量","叫卖序号"]
    buy  = trades[trades["叫买序号"]  != "0"][buy_cols].copy()
    sell = trades[trades["叫卖序号"] != "0"][sell_cols].copy()
    buy .rename(columns={"叫买序号" :"交易所委托号"}, inplace=True)
    sell.rename(columns={"叫卖序号":"交易所委托号"}, inplace=True)
    buy ["方向_事件"]  = "B"
    sell["方向_事件"] = "S"
    events = pd.concat([buy, sell], ignore_index=True)
    events["事件类型"] = np.where(events["成交代码"] == "C", "撤单", "成交")
    events["交易所委托号"] = events["交易所委托号"].str.lstrip("0")

    # 0) 合并委托 + 成交
    merged = events.merge(
        orders,
        on="交易所委托号",
        how="left",
        suffixes=("_事件","_委托")
    )

    # 时间列
    fmt = "%Y%m%d%H%M%S%f"
    merged["委托_datetime"] = pd.to_datetime(
        merged["自然日_委托"].str.zfill(8) + merged["时间_委托"].str.zfill(9),
        errors="coerce", format=fmt)
    merged["事件_datetime"] = pd.to_datetime(
        merged["自然日_事件"].str.zfill(8) + merged["时间_事件"].str.zfill(9),
        errors="coerce", format=fmt)

    # 1) 读取并重采样当日行情
    quote_csv = order_csv.with_name("行情.csv")      # 与逐笔委托同目录
    quotes = prepare_quotes(quote_csv)

    # 2) 贴最近 ≤ t 的买一/卖一
    merged = merged[merged["委托_datetime"].notna()].copy()
    merged.sort_values("委托_datetime", inplace=True)
    merged = pd.merge_asof(
        merged, quotes,
        left_on="委托_datetime", right_on="quote_dt",
        direction="backward",
        tolerance=pd.Timedelta("100ms")              # 容忍 100 ms
    )

    merged["存活时间_ms"] = (
        merged["事件_datetime"] - merged["委托_datetime"]
    ).dt.total_seconds() * 1_000

    merged = merged.loc[:, ~merged.columns.str.contains("^Unnamed")]
    return merged

def merge_one_day(date_dir: Path, watch_set: set, out_root: Path):
    """
    处理单日：date_dir = base_data/YYYYMMDD
    输出 event_stream/YYYYMMDD/委托事件流.csv
    """
    inner = date_dir / date_dir.name
    if not inner.exists():
        inner = date_dir
    dfs = []
    
    # 从目录名提取日期
    trading_date = date_dir.name
    
    for stk_dir in sorted(inner.iterdir()):
        if watch_set and stk_dir.name not in watch_set:
            continue
        o_csv = stk_dir / "逐笔委托.csv"
        t_csv = stk_dir / "逐笔成交.csv"
        if not (o_csv.exists() and t_csv.exists()):
            continue
        df = merge_one_stock(o_csv, t_csv)
        df["ticker"] = stk_dir.name
        
        # 添加统一的交易日期字段，确保数据一致性
        df["自然日"] = trading_date  # 统一的日期字段，用于后续主键
        
        # 数据质量检查：确保日期一致性
        if "自然日_委托" in df.columns:
            inconsistent_dates = df[df["自然日_委托"] != trading_date]
            if not inconsistent_dates.empty:
                print(f"  ⚠️  Warning: Found {len(inconsistent_dates)} records with inconsistent dates in {stk_dir.name}")
        
        dfs.append(df)

    if not dfs:
        print(f"  × {date_dir.name}: no target-stock data")
        return

    out_dir = out_root / date_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "委托事件流.csv"
    
    # 合并所有股票数据
    final_df = pd.concat(dfs, ignore_index=True)
    
    # 重新排列列顺序，将主键字段放在前面
    key_cols = ["自然日", "ticker", "交易所委托号"]
    other_cols = [col for col in final_df.columns if col not in key_cols]
    final_df = final_df[key_cols + other_cols]
    
    final_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ {date_dir.name}: saved → {out_file.name}  ({len(final_df):,} rows)")
    print(f"    📊 Stocks: {final_df['ticker'].nunique()}, Orders: {final_df['交易所委托号'].nunique():,}")

# ────────────────────────── 主流程 ──────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
        help="base_data 根目录")
    parser.add_argument("--tickers", nargs="*", default=None,
        help="股票白名单，留空=全部")
    args = parser.parse_args()

    base = Path(args.root)
    if not base.exists():
        sys.exit(f"[ERROR] 根目录不存在: {base}")
    watch_set = set(args.tickers) if args.tickers else set()

    out_root = base.parent / "event_stream"
    out_root.mkdir(exist_ok=True)

    # 找到形如 8 位日期的子目录
    date_dirs = [p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)]
    if not date_dirs:
        sys.exit("[ERROR] 未找到任何日期目录 (YYYYMMDD)")

    for d in sorted(date_dirs):
        print(f"▶ merging {d.name}")
        merge_one_day(d, watch_set, out_root)

if __name__ == "__main__":
    main()


"""
python scripts/data_process/merge_order_trade.py \
  --root "/obs/users/fenglang/general/Spoofing Detect/data/base_data" \
  --tickers 000989.SZ 300233.SZ


"""