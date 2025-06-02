 # -*- coding: utf-8 -*-
"""
merge_order_trade.py  —— 逐笔委托 × 逐笔成交 整合脚本
执行：python merge_order_trade.py
"""
import pandas as pd
import numpy as np
from pathlib import Path


def read_csv_auto(path: Path) -> pd.DataFrame:
    """优先尝试 GBK，再尝试 UTF-8 读取 CSV。"""
    for enc in ('gbk', 'utf-8'):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f'无法以 GBK/UTF-8 解码 {path}')


def preprocess_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """拆分买/卖两侧，生成事件流长表"""
    trades = trades.copy()
    # -------- 拆 buy side --------
    buy = trades.loc[trades[COLUMN_MAP['buy_seq']] != '0',
                     [COLUMN_MAP[k] for k in ('trade_day','trade_time',
                                              'trade_code','price','qty','buy_seq')]].copy()
    buy.rename(columns={COLUMN_MAP['buy_seq']: COLUMN_MAP['order_id']}, inplace=True)
    buy['方向'] = 'B'
    
    # -------- 拆 sell side -------
    sell = trades.loc[trades[COLUMN_MAP['sell_seq']] != '0',
                      [COLUMN_MAP[k] for k in ('trade_day','trade_time',
                                               'trade_code','price','qty','sell_seq')]].copy()
    sell.rename(columns={COLUMN_MAP['sell_seq']: COLUMN_MAP['order_id']}, inplace=True)
    sell['方向'] = 'S'
    
    events = pd.concat([buy, sell], ignore_index=True)
    events['事件类型'] = np.where(events[COLUMN_MAP['trade_code']] == 'C', '撤单', '成交')
    return events


def merge_orders_events(orders: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """联结委托表与事件表，补充时间戳并计算存活时长（毫秒）"""
    orders[COLUMN_MAP['order_id']] = orders[COLUMN_MAP['order_id']].astype(str)
    events[COLUMN_MAP['order_id']] = events[COLUMN_MAP['order_id']].astype(str)
    
    df = events.merge(orders, on=COLUMN_MAP['order_id'], how='left',
                      suffixes=('_事件','_委托'))
    
    # ---> 构造统一的 datetime
    fmt = '%Y%m%d%H%M%S%f'  # 例：20250506915348123
    df['委托_datetime'] = pd.to_datetime(
        df[COLUMN_MAP['order_day'] + '_委托'].fillna(df[COLUMN_MAP['trade_day'] + '_事件']) # 容错
        .astype(str).str.zfill(8) +
        df[COLUMN_MAP['order_time'] + '_委托'].astype(str).str.zfill(9),
        format=fmt, errors='coerce')
    
    df['事件_datetime'] = pd.to_datetime(
        df[COLUMN_MAP['trade_day'] + '_事件'].astype(str).str.zfill(8) +
        df[COLUMN_MAP['trade_time'] + '_事件'].astype(str).str.zfill(9),
        format=fmt, errors='coerce')
    
    df['存活时间_ms'] = (df['事件_datetime'] - df['委托_datetime']).dt.total_seconds() * 1_000
    return df


def main(folder='.'):
    folder = Path(folder)
    orders = read_csv_auto(folder / FILE_ORDER)
    trades = read_csv_auto(folder / FILE_TRADE)

    # ----------------- 处理 -----------------
    events = preprocess_trades(trades)
    combined = merge_orders_events(orders, events)

    combined.to_csv(folder / OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f'✔️ 已生成：{OUTPUT_FILE}（{len(combined):,} 行）')



# ========= 1. 配置区域（如字段名有差异，请修改这里） ================
COLUMN_MAP = {
    # 逐笔委托表：订单“出生信息”
    'order_id'      : '委托代码',   # 交易所委托号
    'order_day'     : '自然日',     # 交易日期（YYYYMMDD）
    'order_time'    : '时间',       # 委托时间（hhmmssmmm）
    
    # 逐笔成交表：订单“事件流”
    'trade_day'     : '自然日',
    'trade_time'    : '时间',
    'trade_code'    : '成交代码',   # 0=撮合成交，C=撤单
    'price'         : '成交价格',
    'qty'           : '成交数量',
    'buy_seq'       : '叫买序号',
    'sell_seq'      : '叫卖序号',
}

FILE_ORDER = '逐笔委托.csv'
FILE_TRADE = '逐笔成交.csv'
OUTPUT_FILE = '委托事件流.csv'

data_folder = '/home/ma-user/code/fenglang/Spoofing Detect/data/exmaple'

# ===============================================================


if __name__ == '__main__':
    main(data_folder)
