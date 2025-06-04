以下示例以最新的根目录为基准，说明“解压后原始 CSV 文件”的存放结构：

```
/obs/users/fenglang/general/Spoofing Detect/data/base_data/
├── 20250301/
│   └── 20250301/
│       ├── 000989.SZ/
│       │   ├── 行情.csv
│       │   ├── 逐笔成交.csv
│       │   └── 逐笔委托.csv
│       └── 300233.SZ/
│           ├── 行情.csv
│           ├── 逐笔成交.csv
│           └── 逐笔委托.csv
│
├── 20250302/
│   └── 20250302/
│       ├── 000989.SZ/…
│       └── 300233.SZ/…
│
├── 20250303/
│   └── 20250303/
│       ├── 000989.SZ/
│       │   ├── 行情.csv           ← 例如：
│       │   ├── 逐笔成交.csv       “/obs/users/fenglang/general/Spoofing Detect/data/base_data/
│       │   └── 逐笔委托.csv       20250303/20250303/000989.SZ/委托.csv”
│       └── 300233.SZ/
│           ├── 行情.csv
│           ├── 逐笔成交.csv
│           └── 逐笔委托.csv
│
└── …（其余日期，目录结构相同）
```

* **第一层目录**（如 `20250303/`）对应压缩包所在的日期。
* **第二层目录**（与第一层同名）是压缩包内部的顶层文件夹。
* **第三层目录**（股票代码，如 `000989.SZ`）下分别存放当日该股票的三个文件：

  * `行情.csv`
  * `逐笔成交.csv`
  * `逐笔委托.csv`

完整示例路径：

```
/obs/users/fenglang/general/Spoofing Detect/data/base_data/20250303/20250303/000989.SZ/行情.csv
/obs/users/fenglang/general/Spoofing Detect/data/base_data/20250303/20250303/300233.SZ/逐笔委托.csv
```
