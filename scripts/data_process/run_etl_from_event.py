#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_etl_from_event.py
─────────────────────
1. 输入  : event_stream/<YYYYMMDD>/委托事件流.csv
2. 输出  : features_select/X_<YYYYMMDD>.parquet
           labels_select/labels_<YYYYMMDD>.parquet
3. 调用  :
      python run_etl_from_event.py \
          --root /obs/.../event_stream \
          --tickers 000989.SZ 300233.SZ \
          --r1_ms 50 --r2_ms 1000 --r2_mult 4.0 \
          --backend polars  # 新增：选择计算后端
"""

import pandas as pd, numpy as np
import polars as pl
from pathlib import Path
import argparse, re, sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich import print as rprint
from concurrent.futures import ProcessPoolExecutor
import os

# Configure console
console = Console()

# ──────────────────────────── config ────────────────────────────
STRICT = True  # True = 缺列立即报错；False = 自动填 NaN

# 设置 Polars 线程数
os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())

# ──────────────────────────── util ────────────────────────────
def require_cols(df: pd.DataFrame, cols: list[str]):
    """严格模式：保证必须列全部存在，否则抛错"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要行情列：{missing}")

def calc_realtime_features(df: pd.DataFrame) -> pd.DataFrame:
    """实时可观测特征列"""
    # ---------- 0. 必要列校验 ----------
    needed = ["申买价1", "申卖价1", "前收盘"]
    if STRICT:
        require_cols(df, needed)

    # ---------- 1. 行情原始列 ----------
    df["bid1"] = df.get("申买价1")
    df["ask1"] = df.get("申卖价1")
    df["prev_close"] = df.get("前收盘")

    # ---------- 2. 价格派生 ----------
    df["is_buy"] = (df["方向_委托"] == "买").astype(int)
    df["mid_price"] = (df["bid1"] + df["ask1"]) / 2
    df["spread"] = df["ask1"] - df["bid1"]
    df["delta_mid"] = df["委托价格"] - df["mid_price"]
    df["pct_spread"] = (df["delta_mid"].abs() / df["spread"]).replace([np.inf,-np.inf],0).fillna(0)
    df["log_qty"] = np.log1p(df["委托数量"])

    df.sort_values("委托_datetime", inplace=True)
    # 订单数统计：使用 count() 而不是 sum()
    df["orders_100ms"] = (
        df.rolling("100ms", on="委托_datetime")["委托_datetime"]
          .count().shift(1).fillna(0))
    
    # 撤单统计：先转换为 0/1 再 sum
    df["is_cancel"] = (df["事件类型"] == "撤单").astype(int)
    df["cancels_5s"] = (
        df.rolling("5s", on="委托_datetime")["is_cancel"]
          .sum().shift(1).fillna(0))

    sec = (
        df["委托_datetime"]
        - df["委托_datetime"].dt.normalize()
        - pd.Timedelta("09:30:00")
    ).dt.total_seconds()
    df["time_sin"] = np.sin(2*np.pi*sec/(4.5*3600))
    df["time_cos"] = np.cos(2*np.pi*sec/(4.5*3600))

    df["in_auction"] = (
        (df["委托_datetime"].dt.time.between(pd.to_datetime("09:15").time(),
                                            pd.to_datetime("09:30").time(), "left"))
      | (df["委托_datetime"].dt.time.between(pd.to_datetime("14:57").time(),
                                            pd.to_datetime("15:00").time(), "both"))
    ).astype(int)
    df["price_dev_prevclose"] = (
        (df["委托价格"] - df["prev_close"]) / df["prev_close"]).fillna(0)

    return df

def apply_label_rules(df: pd.DataFrame, r1_ms: int, r2_ms: int, r2_mult: float):
    """
    修正版标签规则：
    - R1: 快速撤单（存活时间 < r1_ms 且事件类型为撤单）
    - R2: 偏离市场中心价（存活时间 < r2_ms 且价格偏离 >= r2_mult * spread）
    """
    # R1: 快速撤单（事件类型为撤单且存活时间很短）
    df["flag_R1"] = (df["存活时间_ms"] < r1_ms) & (df["事件类型"] == "撤单")

    # R2: 价格偏离（存活时间短且价格偏离大）
    # NaN spread 在 R2 里应视为"不满足" → 用 +∞ 占位
    safe_spread = df["spread"].fillna(np.inf)
    df["flag_R2"] = (
        (df["存活时间_ms"] < r2_ms) &
        (df["delta_mid"].abs() >= r2_mult * safe_spread)
    )
    df["y_label"] = (df["flag_R1"] & df["flag_R2"]).astype(int)
    
    return df

def calc_realtime_features_polars(df: pl.LazyFrame) -> pl.LazyFrame:
    """使用 Polars Lazy API 计算实时可观测特征"""
    # 确保时间列是 ns 精度，用于窗口对齐
    df = df.with_columns([
        pl.col("委托_datetime").cast(pl.Datetime("ns")).alias("ts_ns"),
        pl.col("事件_datetime").cast(pl.Datetime("ns")).alias("event_ts_ns"),
    ])
    
    # 基础特征计算
    df = df.with_columns([
        # 行情原始列
        pl.col("申买价1").alias("bid1"),
        pl.col("申卖价1").alias("ask1"),
        pl.col("前收盘").alias("prev_close"),
        
        # 价格派生
        (pl.col("方向_委托") == "买").cast(pl.Int8).alias("is_buy"),
        ((pl.col("申买价1") + pl.col("申卖价1")) / 2).alias("mid_price"),
        (pl.col("申卖价1") - pl.col("申买价1")).alias("spread"),
        (pl.col("委托价格") - (pl.col("申买价1") + pl.col("申卖价1")) / 2).alias("delta_mid"),
        (pl.col("delta_mid").abs() / pl.col("spread")).fill_null(0).replace([float("inf"), float("-inf")], 0).alias("pct_spread"),
        pl.col("委托数量").log1p().alias("log_qty"),
        
        # 撤单标志
        (pl.col("事件类型") == "撤单").cast(pl.Int8).alias("is_cancel"),
        
        # 时间特征
        ((pl.col("委托_datetime") - pl.col("委托_datetime").dt.normalize() - pl.duration(hours=9, minutes=30))
         .dt.total_seconds()
         .map_elements(lambda x: np.sin(2*np.pi*x/(4.5*3600))).alias("time_sin")),
        ((pl.col("委托_datetime") - pl.col("委托_datetime").dt.normalize() - pl.duration(hours=9, minutes=30))
         .dt.total_seconds()
         .map_elements(lambda x: np.cos(2*np.pi*x/(4.5*3600))).alias("time_cos")),
        
        # 集合竞价时段
        (pl.col("委托_datetime").dt.time().is_between(
            pl.time(9,15), pl.time(9,30), "left") |
         pl.col("委托_datetime").dt.time().is_between(
            pl.time(14,57), pl.time(15,0), "both")
        ).cast(pl.Int8).alias("in_auction"),
        
        # 价格偏离
        ((pl.col("委托价格") - pl.col("前收盘")) / pl.col("前收盘")).fill_null(0).alias("price_dev_prevclose"),
    ])
    
    # 滚动窗口特征
    # 1. 100ms 订单计数
    orders_100ms = (
        df.select(["ts_ns"])
        .with_columns(pl.lit(1).alias("is_order"))
        .groupby_dynamic(
            index_column="ts_ns",
            every="100ms",
            period="100ms",
            closed="right"
        )
        .agg(pl.col("is_order").count().alias("orders_100ms"))
    )
    
    # 2. 5s 撤单计数
    cancels_5s = (
        df.select(["ts_ns", "is_cancel"])
        .groupby_dynamic(
            index_column="ts_ns",
            every="1s",
            period="5s",
            closed="right"
        )
        .agg(pl.col("is_cancel").sum().alias("cancels_5s"))
    )
    
    # 合并滚动窗口特征
    df = (
        df.join_asof(orders_100ms, on="ts_ns", strategy="backward", tolerance="100ms")
        .join_asof(cancels_5s, on="ts_ns", strategy="backward", tolerance="5s")
    )
    
    return df

def process_one_day_polars(evt_csv: Path, feat_dir: Path, lbl_dir: Path,
                          watch: set, r1_ms: int, r2_ms: int, r2_mult: float):
    """使用 Polars 处理单日数据"""
    date_str = evt_csv.parent.name
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # 任务1: 加载数据
        load_task = progress.add_task(f"[cyan]{date_str}[/cyan] Loading CSV...", total=100)
        
        # 使用 Polars 读取 CSV
        df = pl.scan_csv(evt_csv, 
                        try_parse_dates=True,
                        infer_schema_length=10000)
        progress.update(load_task, advance=100)
        
        if watch:
            df = df.filter(pl.col("ticker").is_in(list(watch)))
        
        # 添加日期列
        df = df.with_columns(pl.lit(date_str).alias("自然日"))
        
        # 任务2: 计算特征
        feat_task = progress.add_task(f"[green]{date_str}[/green] Computing features...", total=100)
        df = calc_realtime_features_polars(df)
        progress.update(feat_task, advance=50)
        
        # 应用标签规则
        df = df.with_columns([
            # R1: 快速撤单
            ((pl.col("存活时间_ms") < r1_ms) & 
             (pl.col("事件类型") == "撤单")).cast(pl.Int8).alias("flag_R1"),
            
            # R2: 价格偏离
            ((pl.col("存活时间_ms") < r2_ms) & 
             (pl.col("delta_mid").abs() >= r2_mult * pl.col("spread").fill_null(float("inf")))).cast(pl.Int8).alias("flag_R2"),
        ])
        
        # 计算最终标签
        df = df.with_columns(
            (pl.col("flag_R1") & pl.col("flag_R2")).cast(pl.Int8).alias("y_label")
        )
        progress.update(feat_task, advance=50)
        
        # 任务3: 按委托聚合
        agg_task = progress.add_task(f"[yellow]{date_str}[/yellow] Aggregating orders...", total=100)
        
        # 按委托分组聚合
        df_orders = (
            df.groupby(["自然日", "ticker", "交易所委托号"])
            .agg([
                # 取第一行特征
                pl.first("bid1", "ask1", "prev_close", "mid_price", "spread", 
                        "delta_mid", "pct_spread", "orders_100ms", "cancels_5s",
                        "log_qty", "time_sin", "time_cos", "in_auction",
                        "price_dev_prevclose", "is_buy", "is_cancel"),
                
                # 聚合统计
                pl.count().alias("total_events"),
                pl.col("成交数量").filter(pl.col("事件类型") == "成交").sum().alias("total_traded_qty"),
                pl.col("事件类型").filter(pl.col("事件类型") == "成交").count().alias("num_trades"),
                pl.col("事件类型").filter(pl.col("事件类型") == "撤单").count().alias("num_cancels"),
                
                # 委托存活时间
                (pl.col("event_ts_ns").max() - pl.col("ts_ns").first()).dt.milliseconds().alias("final_survival_time_ms"),
                
                # 是否完全成交
                ((pl.col("成交数量").filter(pl.col("事件类型") == "成交").sum() >= 
                  pl.col("委托数量").first() * 0.99)).cast(pl.Int8).alias("is_fully_filled"),
                
                # 标签
                pl.col("y_label").max().alias("y_label")
            ])
        )
        progress.update(agg_task, advance=100)
        
        # 任务4: 保存数据
        save_task = progress.add_task(f"[magenta]{date_str}[/magenta] Saving files...", total=100)
        
        # 收集结果并保存
        df_orders = df_orders.collect(streaming=True)
        
        if df_orders.is_empty():
            return 0, 0
        
        feat_cols = [
            "自然日", "交易所委托号", "ticker",
            "bid1", "ask1", "prev_close",
            "mid_price", "spread", "delta_mid", "pct_spread",
            "orders_100ms", "cancels_5s", "log_qty",
            "time_sin", "time_cos", "in_auction",
            "price_dev_prevclose", "is_buy", "is_cancel",
            "total_events", "total_traded_qty", "num_trades", "num_cancels",
            "final_survival_time_ms", "is_fully_filled"
        ]
        
        feat_dir.mkdir(exist_ok=True); lbl_dir.mkdir(exist_ok=True)
        progress.update(save_task, advance=30)
        
        df_orders.select(feat_cols).write_parquet(
            feat_dir / f"X_{date_str}.parquet",
            compression="snappy"
        )
        progress.update(save_task, advance=35)
        
        df_orders.select(["自然日", "交易所委托号", "ticker", "y_label"]).write_parquet(
            lbl_dir / f"labels_{date_str}.parquet",
            compression="snappy"
        )
        progress.update(save_task, advance=35)
        
        return len(df_orders), int(df_orders["y_label"].sum())

def process_one_day(evt_csv: Path, feat_dir: Path, lbl_dir: Path,
                    watch: set, r1_ms: int, r2_ms: int, r2_mult: float,
                    backend: str = "pandas"):
    """根据后端选择处理函数"""
    if backend == "polars":
        return process_one_day_polars(evt_csv, feat_dir, lbl_dir, watch, r1_ms, r2_ms, r2_mult)
    else:
        # 原有的 Pandas 处理逻辑
        date_str = evt_csv.parent.name
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # 任务1: 加载数据
            load_task = progress.add_task(f"[cyan]{date_str}[/cyan] Loading CSV...", total=100)
            
            # 设置 low_memory=False 避免混合类型警告
            df = pd.read_csv(evt_csv, 
                             parse_dates=["委托_datetime","事件_datetime"],
                             low_memory=False)
            progress.update(load_task, advance=100)
            
            if watch:
                df = df[df["ticker"].isin(watch)]
            if df.empty:
                return 0, 0

            # 添加日期列（从文件路径提取）
            df["自然日"] = date_str  # 添加交易日期列
            
            # 任务2: 计算基础特征
            feat_task = progress.add_task(f"[green]{date_str}[/green] Computing features...", total=100)
            df = calc_realtime_features(df)
            progress.update(feat_task, advance=50)
            
            df = apply_label_rules(df, r1_ms, r2_ms, r2_mult)
            progress.update(feat_task, advance=50)
            
            # 任务3: 按委托聚合数据
            # 按委托分组聚合
            order_groups = df.groupby(["自然日", "ticker", "交易所委托号"])
            total_orders = len(order_groups)
            
            agg_task = progress.add_task(f"[yellow]{date_str}[/yellow] Aggregating {total_orders:,} orders...", total=total_orders)
            
            order_features = []
            
            for i, ((date, ticker, order_id), group) in enumerate(order_groups):
                # 取委托时刻的特征（第一行，因为按委托时间排序）
                first_row = group.iloc[0].copy()
                
                # 聚合标签：如果任何一个事件触发了标签，则整个委托被标记为正例
                aggregated_label = group["y_label"].max()  # 任何事件触发 -> 整个委托为正例
                
                # 添加聚合统计特征
                first_row["total_events"] = len(group)  # 该委托总共有多少个事件
                first_row["total_traded_qty"] = group[group["事件类型"] == "成交"]["成交数量"].astype(float).sum()  # 总成交量
                first_row["num_trades"] = (group["事件类型"] == "成交").sum()  # 成交次数
                first_row["num_cancels"] = (group["事件类型"] == "撤单").sum()  # 撤单次数
                
                # 委托存活时间：最后一个事件的时间 - 委托时间
                first_row["final_survival_time_ms"] = (group["事件_datetime"].max() - group["委托_datetime"].iloc[0]).total_seconds() * 1000
                
                # 是否完全成交
                total_order_qty = float(first_row["委托数量"])
                first_row["is_fully_filled"] = int(first_row["total_traded_qty"] >= total_order_qty * 0.99)  # 允许0.01的舍入误差
                
                # 更新标签
                first_row["y_label"] = aggregated_label
                
                order_features.append(first_row)
                
                # 更新进度（每100个委托更新一次以提高性能）
                if i % 100 == 0 or i == total_orders - 1:
                    progress.update(agg_task, completed=i + 1)
            
            # 任务4: 保存数据
            save_task = progress.add_task(f"[magenta]{date_str}[/magenta] Saving files...", total=100)
            
            # 转换为DataFrame
            df_orders = pd.DataFrame(order_features)
            
            if df_orders.empty:
                return 0, 0
            
            feat_cols = [
                "自然日",  # 添加日期作为主键的一部分
                "交易所委托号","ticker",
                # --- 行情原始列 ---
                "bid1","ask1","prev_close",
                # --- 价格派生 ---
                "mid_price","spread","delta_mid","pct_spread",
                # --- 订单流特征 ---
                "orders_100ms","cancels_5s","log_qty",
                # --- 时间特征 ---
                "time_sin","time_cos","in_auction",
                "price_dev_prevclose",
                # --- 方向 ---
                "is_buy",
                # --- 撤单标志 ---
                "is_cancel",
                # --- 新增聚合特征 ---
                "total_events", "total_traded_qty", "num_trades", "num_cancels",
                "final_survival_time_ms", "is_fully_filled"
            ]
            
            feat_dir.mkdir(exist_ok=True); lbl_dir.mkdir(exist_ok=True)
            progress.update(save_task, advance=30)
            
            df_orders[feat_cols].to_parquet(feat_dir / f"X_{date_str}.parquet", index=False)
            progress.update(save_task, advance=35)
            
            df_orders[["自然日","交易所委托号","ticker","y_label"]].to_parquet(  # 添加日期到标签文件
                lbl_dir / f"labels_{date_str}.parquet", index=False)
            progress.update(save_task, advance=35)
            
            return len(df_orders), int(df_orders["y_label"].sum())

# ───────────────────────────── main ────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="event_stream 根目录")
    parser.add_argument("--tickers", nargs="*", default=None, help="股票白名单")
    parser.add_argument("--r1_ms", type=int, default=50)
    parser.add_argument("--r2_ms", type=int, default=1000)
    parser.add_argument("--r2_mult", type=float, default=4.0)
    parser.add_argument("--backend", choices=["pandas", "polars"], default="polars",
                      help="选择计算后端：pandas 或 polars")
    parser.add_argument("--max_workers", type=int, default=None,
                      help="进程池最大工作进程数（仅 polars 后端有效）")
    args = parser.parse_args()

    evt_root = Path(args.root)
    parent   = evt_root.parent
    feat_dir = parent / "features_select"
    lbl_dir  = parent / "labels_select"
    watch    = set(args.tickers) if args.tickers else set()

    date_dirs = [p for p in evt_root.iterdir()
                 if p.is_dir() and re.fullmatch(r"\d{8}", p.name)]
    if not date_dirs:
        sys.exit("[ERR] no date folders under event_stream")

    # 显示总体进度
    console.print(f"\n[bold green]🚀 Starting ETL Pipeline[/bold green]")
    console.print(f"[dim]Root: {evt_root}[/dim]")
    console.print(f"[dim]Backend: {args.backend}[/dim]")
    console.print(f"[dim]Tickers: {list(watch) if watch else 'All'}[/dim]")
    console.print(f"[dim]Found {len(date_dirs)} date directories[/dim]\n")
    
    total_rows = 0
    total_positives = 0
    successful_days = 0
    
    if args.backend == "polars" and args.max_workers:
        # 使用进程池并行处理多个日期
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for d in sorted(date_dirs):
                evt_csv = d / "委托事件流.csv"
                if not evt_csv.exists():
                    rprint(f"[red]× {d.name}: missing csv[/red]")
                    continue
                    
                futures.append(
                    executor.submit(
                        process_one_day,
                        evt_csv, feat_dir, lbl_dir,
                        watch, args.r1_ms, args.r2_ms, args.r2_mult,
                        args.backend
                    )
                )
            
            for d, future in zip(sorted(date_dirs), futures):
                try:
                    rows, pos = future.result()
                    if rows:
                        rprint(f"[green]✓ {d.name}: orders={rows:,}  positive={pos}[/green]")
                        total_rows += rows
                        total_positives += pos
                        successful_days += 1
                    else:
                        rprint(f"[yellow]× {d.name}: no watch-list rows[/yellow]")
                except Exception as e:
                    rprint(f"[red]× {d.name}: ERROR - {str(e)}[/red]")
    else:
        # 串行处理
        for d in sorted(date_dirs):
            evt_csv = d / "委托事件流.csv"
            if not evt_csv.exists():
                rprint(f"[red]× {d.name}: missing csv[/red]")
                continue
                
            try:
                rows, pos = process_one_day(evt_csv, feat_dir, lbl_dir,
                                          watch, args.r1_ms, args.r2_ms, args.r2_mult,
                                          args.backend)
                if rows:
                    rprint(f"[green]✓ {d.name}: orders={rows:,}  positive={pos}[/green]")
                    total_rows += rows
                    total_positives += pos
                    successful_days += 1
                else:
                    rprint(f"[yellow]× {d.name}: no watch-list rows[/yellow]")
            except Exception as e:
                rprint(f"[red]× {d.name}: ERROR - {str(e)}[/red]")
                continue
    
    # 显示总结
    console.print(f"\n[bold cyan]📊 ETL Pipeline Summary[/bold cyan]")
    console.print(f"  Backend: {args.backend}")
    console.print(f"  Processed days: {successful_days}/{len(date_dirs)}")
    console.print(f"  Total orders: {total_rows:,}")
    console.print(f"  Positive labels: {total_positives:,}")
    if total_rows > 0:
        positive_rate = total_positives / total_rows * 100
        console.print(f"  Positive rate: {positive_rate:.4f}%")
    console.print(f"[bold green]✅ ETL Pipeline completed![/bold green]\n")

if __name__ == "__main__":
    main()

"""
# 使用 Polars 后端（推荐）
python scripts/data_process/run_etl_from_event.py \
    --root "/obs/users/fenglang/general/Spoofing Detect/data/event_stream" \
    --tickers 000989.SZ 300233.SZ \
    --r1_ms 50 --r2_ms 1000 --r2_mult 4.0 \
    --backend polars \
    --max_workers 4  # 可选：进程级并行

# 使用 Pandas 后端（兼容）
python scripts/data_process/run_etl_from_event.py \
    --root "/obs/users/fenglang/general/Spoofing Detect/data/event_stream" \
    --tickers 000989.SZ 300233.SZ \
    --r1_ms 50 --r2_ms 1000 --r2_mult 4.0 \
    --backend pandas
"""