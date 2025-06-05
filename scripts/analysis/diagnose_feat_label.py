#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick diagnosis of generated features & labels
------------------------------------------------
Author : YourName
Usage  :
    python diagnose_feat_label.py --root "/your/event_stream_root" --month "202504" --topn 20
"""
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

class DataDiagnostics:
    """Class for diagnosing feature and label data"""
    
    def __init__(self, root_path: str, month_pattern: str = "", top_n: int = 20, save_stock_analysis: bool = False):
        self.root = Path(root_path)
        self.month_pattern = f"*{month_pattern}*" if month_pattern else "*"
        self.top_n = top_n
        self.save_stock_analysis = save_stock_analysis
        self._validate_root_directory()
    
    def _validate_root_directory(self) -> None:
        """Validate that the root directory exists and contains required subdirectories"""
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root}")
        
        features_dir = self.root / "features_select"
        labels_dir = self.root / "labels_select"
        
        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory does not exist: {features_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")
    
    def _find_parquet_files(self) -> Tuple[List[Path], List[Path]]:
        """Find and validate parquet files"""
        feat_files = sorted((self.root / "features_select").glob(f"{self.month_pattern}.parquet"))
        lab_files = sorted((self.root / "labels_select").glob(f"{self.month_pattern}.parquet"))
        
        if not feat_files:
            raise FileNotFoundError(f"No feature files found matching pattern: {self.month_pattern}")
        if not lab_files:
            raise FileNotFoundError(f"No label files found matching pattern: {self.month_pattern}")
            
        logger.info(f"Found {len(feat_files)} feature files and {len(lab_files)} label files")
        return feat_files, lab_files
    
    def _load_parquet_files(self, files: List[Path], data_type: str) -> pd.DataFrame:
        """Load and concatenate parquet files with progress tracking"""
        if not files:
            raise ValueError(f"No {data_type} files to load")
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[progress.description]Loading {data_type} files..."),
            console=console
        ) as progress:
            task = progress.add_task(f"Loading {data_type}", total=len(files))
            
            dataframes = []
            for file_path in files:
                try:
                    df = pd.read_parquet(file_path)
                    dataframes.append(df)
                    progress.advance(task)
                    logger.debug(f"Loaded {file_path}: {df.shape}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    raise
        
        result_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Concatenated {data_type} shape: {result_df.shape}")
        return result_df
    
    def _merge_data(self, df_feat: pd.DataFrame, df_lab: pd.DataFrame) -> pd.DataFrame:
        """Merge feature and label dataframes using composite key (date + ticker + order_id)"""
        # Define potential key columns and their suffixes
        key_triplets = [
            # (date_col, ticker_col, order_id_col)
            ("自然日", "ticker", "交易所委托号"),
            ("date", "ticker", "交易所委托号"),
            ("trading_date", "ticker", "order_id"),
            ("日期", "ticker", "委托号"),
            ("自然日", "stock_code", "交易所委托号"),
        ]
        
        # Try to find matching key columns with suffixes
        merge_keys = []
        for date_base, ticker_base, order_id_base in key_triplets:
            # Check for suffix patterns
            date_feat = f"{date_base}_feat" if f"{date_base}_feat" in df_feat.columns else date_base
            date_lab = f"{date_base}_lab" if f"{date_base}_lab" in df_lab.columns else date_base
            ticker_feat = f"{ticker_base}_feat" if f"{ticker_base}_feat" in df_feat.columns else ticker_base
            ticker_lab = f"{ticker_base}_lab" if f"{ticker_base}_lab" in df_lab.columns else ticker_base
            order_id_feat = f"{order_id_base}_feat" if f"{order_id_base}_feat" in df_feat.columns else order_id_base
            order_id_lab = f"{order_id_base}_lab" if f"{order_id_base}_lab" in df_lab.columns else order_id_base
            
            if (date_feat in df_feat.columns and date_lab in df_lab.columns and
                ticker_feat in df_feat.columns and ticker_lab in df_lab.columns and
                order_id_feat in df_feat.columns and order_id_lab in df_lab.columns):
                merge_keys = [
                    (date_feat, date_lab), 
                    (ticker_feat, ticker_lab), 
                    (order_id_feat, order_id_lab)
                ]
                logger.info(f"Found composite merge keys: {date_feat} + {ticker_feat} + {order_id_feat} <-> {date_lab} + {ticker_lab} + {order_id_lab}")
                break
        
        # Fallback: try without date if date columns not found
        if not merge_keys:
            logger.warning("Date column not found, trying ticker + order_id only")
            key_pairs = [
                # (ticker_col, order_id_col)
                ("ticker", "交易所委托号"),
                ("ticker", "order_id"),
                ("ticker", "委托号"),
                ("stock_code", "交易所委托号"),
            ]
            
            for ticker_base, order_id_base in key_pairs:
                ticker_feat = f"{ticker_base}_feat" if f"{ticker_base}_feat" in df_feat.columns else ticker_base
                ticker_lab = f"{ticker_base}_lab" if f"{ticker_base}_lab" in df_lab.columns else ticker_base
                order_id_feat = f"{order_id_base}_feat" if f"{order_id_base}_feat" in df_feat.columns else order_id_base
                order_id_lab = f"{order_id_base}_lab" if f"{order_id_base}_lab" in df_lab.columns else order_id_base
                
                if (ticker_feat in df_feat.columns and ticker_lab in df_lab.columns and
                    order_id_feat in df_feat.columns and order_id_lab in df_lab.columns):
                    merge_keys = [(ticker_feat, ticker_lab), (order_id_feat, order_id_lab)]
                    logger.warning(f"Using fallback merge keys: {ticker_feat} + {order_id_feat} <-> {ticker_lab} + {order_id_lab}")
                    logger.warning("⚠️  Without date in merge key, there may be duplicate records across different trading days!")
                    break
        
        if merge_keys:
            # Perform merge using composite key
            left_keys = [mk[0] for mk in merge_keys]
            right_keys = [mk[1] for mk in merge_keys]
            
            df_merged = df_feat.merge(
                df_lab,
                left_on=left_keys,
                right_on=right_keys,
                how="inner"
            )
            
            # Check merge quality
            feat_unique = df_feat[left_keys].drop_duplicates().shape[0]
            lab_unique = df_lab[right_keys].drop_duplicates().shape[0]
            merged_unique = df_merged[left_keys].drop_duplicates().shape[0]
            
            logger.info(f"Merge quality - Features: {feat_unique:,}, Labels: {lab_unique:,}, Merged: {merged_unique:,}")
            
            if merged_unique < min(feat_unique, lab_unique) * 0.9:
                logger.warning("Significant data loss during merge. Check data consistency.")
            
            # Verify no duplicates in merged data
            dup_count = df_merged.duplicated(subset=left_keys).sum()
            if dup_count > 0:
                logger.warning(f"Found {dup_count:,} duplicate records after merge!")
                
                if len(merge_keys) == 2:  # Only ticker + order_id
                    console.print(f"\n[bold red]⚠️  Data Quality Issue Detected![/bold red]")
                    console.print(f"[yellow]Found {dup_count:,} duplicate records with same ticker + order_id[/yellow]")
                    console.print(f"[dim]This likely indicates that order IDs repeat across different trading days.[/dim]")
                    console.print(f"[dim]Recommendation: Ensure date column is included in the data pipeline.[/dim]")
            else:
                logger.info("No duplicate records found after merge ✓")
            
        else:
            logger.warning("No suitable composite merge keys found. Using index-based join with suffixes.")
            df_merged = df_feat.join(df_lab, lsuffix='_feat', rsuffix='_lab')
            console.print(f"\n[bold yellow]⚠️  Fallback to index-based join![/bold yellow]")
            console.print(f"[dim]This may not correctly match records. Consider adding proper merge keys.[/dim]")
        
        return df_merged
    
    def _analyze_labels(self, df: pd.DataFrame) -> None:
        """Analyze and display label distribution"""
        # Find label column
        possible_label_cols = ["label", "target", "y", "y_label"] + [col for col in df.columns if "label" in col.lower()]
        label_col = None
        
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        
        if not label_col:
            # Use last column as fallback
            label_col = df.columns[-1]
            logger.warning(f"No obvious label column found. Using last column: {label_col}")
        
        # Create distribution table
        vc = df[label_col].value_counts(dropna=False)
        total = len(df)
        
        table = Table(title=f"Overall Label Distribution ({label_col})")
        table.add_column("Value", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")
        
        for value, count in vc.items():
            percentage = f"{count/total:.2%}"
            table.add_row(str(value), str(count), percentage)
        
        console.print("\n")
        console.print(table)
        
        # Analyze class imbalance
        if len(vc) >= 2:
            min_class_ratio = vc.min() / total
            max_class_ratio = vc.max() / total
            imbalance_ratio = max_class_ratio / min_class_ratio
            
            console.print(f"\n[bold yellow]Class Imbalance Analysis:[/bold yellow]")
            console.print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
            console.print(f"  Minority class ratio: {min_class_ratio:.4%}")
            
            if imbalance_ratio > 100:
                console.print(f"  [bold red]⚠️  SEVERE class imbalance detected![/bold red]")
                console.print(f"  [dim]Recommendations:[/dim]")
                console.print(f"    • Consider SMOTE or other resampling techniques")
                console.print(f"    • Use stratified sampling for train/test split")
                console.print(f"    • Consider focal loss or class weights")
                console.print(f"    • Use precision, recall, F1 instead of accuracy")
            elif imbalance_ratio > 10:
                console.print(f"  [yellow]⚠️  Moderate class imbalance detected[/yellow]")
                console.print(f"  [dim]Consider using stratified sampling and appropriate metrics[/dim]")
    
    def _find_stock_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the stock/ticker column in the dataframe"""
        possible_stock_cols = [
            "ticker", "stock_code", "symbol", "code", "股票代码", "证券代码",
            "ticker_feat", "ticker_lab", "stock_code_feat", "stock_code_lab"
        ]
        
        for col in possible_stock_cols:
            if col in df.columns:
                return col
        
        # Check for columns containing these keywords
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ["ticker", "stock", "symbol", "code"]):
                return col
                
        return None
    
    def _analyze_by_stock(self, df: pd.DataFrame) -> None:
        """Analyze data quality and label distribution by stock"""
        stock_col = self._find_stock_column(df)
        
        if not stock_col:
            console.print(f"\n[yellow]⚠️  Stock column not found. Skipping stock-level analysis.[/yellow]")
            console.print(f"[dim]Available columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}[/dim]")
            return
        
        console.print(f"\n[bold cyan]Stock-Level Analysis (using column: {stock_col}):[/bold cyan]")
        
        # Find label column
        possible_label_cols = ["label", "target", "y", "y_label"] + [col for col in df.columns if "label" in col.lower()]
        label_col = None
        
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        
        if not label_col:
            label_col = df.columns[-1]
        
        # Group by stock and analyze
        stock_groups = df.groupby(stock_col)
        stock_stats = []
        
        for stock, group in stock_groups:
            # Basic stats
            total_records = len(group)
            
            # Label distribution
            label_dist = group[label_col].value_counts()
            positive_ratio = 0
            if len(label_dist) >= 2:
                # Assume binary classification, take the minority class as positive
                if label_dist.min() / total_records < 0.5:
                    positive_ratio = label_dist.min() / total_records
                else:
                    positive_ratio = label_dist.max() / total_records
            elif len(label_dist) == 1:
                positive_ratio = 1.0 if label_dist.index[0] == 1 else 0.0
            
            # Missing value analysis
            numeric_cols = group.select_dtypes(include=['number']).columns
            missing_rate = group[numeric_cols].isna().mean().mean() if len(numeric_cols) > 0 else 0
            
            # Data quality metrics
            duplicates_rate = group.duplicated().sum() / total_records
            
            stock_stats.append({
                'stock': stock,
                'records': total_records,
                'positive_ratio': positive_ratio,
                'missing_rate': missing_rate,
                'duplicates_rate': duplicates_rate,
                'label_diversity': len(label_dist)
            })
        
        # Convert to DataFrame for easier display
        stats_df = pd.DataFrame(stock_stats).sort_values('records', ascending=False)
        
        # Display summary table
        table = Table(title=f"Stock-Level Summary (Top {min(self.top_n, len(stats_df))} by record count)")
        table.add_column("Stock", style="cyan")
        table.add_column("Records", style="magenta")
        table.add_column("Positive %", style="green")
        table.add_column("Missing %", style="yellow") 
        table.add_column("Duplicate %", style="red")
        table.add_column("Label Types", style="blue")
        
        for _, row in stats_df.head(self.top_n).iterrows():
            table.add_row(
                str(row['stock'])[:15],  # Truncate long stock codes
                f"{int(row['records']):,}",
                f"{row['positive_ratio']:.2%}",
                f"{row['missing_rate']:.2%}",
                f"{row['duplicates_rate']:.2%}",
                str(int(row['label_diversity']))
            )
        
        console.print("\n")
        console.print(table)
        
        # Summary statistics across stocks
        console.print(f"\n[bold yellow]Cross-Stock Summary:[/bold yellow]")
        console.print(f"  Total stocks: {len(stats_df)}")
        console.print(f"  Records per stock - Mean: {stats_df['records'].mean():.0f}, Median: {stats_df['records'].median():.0f}")
        console.print(f"  Positive ratio - Mean: {stats_df['positive_ratio'].mean():.2%}, Std: {stats_df['positive_ratio'].std():.2%}")
        console.print(f"  Missing rate - Mean: {stats_df['missing_rate'].mean():.2%}, Max: {stats_df['missing_rate'].max():.2%}")
        
        # Identify problematic stocks
        high_missing_stocks = stats_df[stats_df['missing_rate'] > 0.5]
        high_duplicate_stocks = stats_df[stats_df['duplicates_rate'] > 0.1]
        extreme_imbalance_stocks = stats_df[(stats_df['positive_ratio'] < 0.01) | (stats_df['positive_ratio'] > 0.99)]
        
        if len(high_missing_stocks) > 0:
            console.print(f"\n[bold red]⚠️  Stocks with >50% missing data: {len(high_missing_stocks)}[/bold red]")
            for _, row in high_missing_stocks.head(5).iterrows():
                console.print(f"    {row['stock']}: {row['missing_rate']:.1%} missing")
        
        if len(high_duplicate_stocks) > 0:
            console.print(f"\n[bold red]⚠️  Stocks with >10% duplicates: {len(high_duplicate_stocks)}[/bold red]")
            for _, row in high_duplicate_stocks.head(5).iterrows():
                console.print(f"    {row['stock']}: {row['duplicates_rate']:.1%} duplicates")
        
        if len(extreme_imbalance_stocks) > 0:
            console.print(f"\n[bold red]⚠️  Stocks with extreme label imbalance: {len(extreme_imbalance_stocks)}[/bold red]")
            for _, row in extreme_imbalance_stocks.head(5).iterrows():
                console.print(f"    {row['stock']}: {row['positive_ratio']:.1%} positive ratio")
        
        # Save detailed stock analysis if requested
        if self.save_stock_analysis:
            month_str = self.month_pattern.replace('*', '') or 'all'
            output_path = self.root / f"stock_analysis_{month_str}.csv"
            stats_df.to_csv(output_path, index=False)
            console.print(f"\n[dim]Detailed stock analysis saved to: {output_path}[/dim]")
    
    def _analyze_stock_feature_quality(self, df: pd.DataFrame) -> None:
        """Analyze feature quality across different stocks"""
        stock_col = self._find_stock_column(df)
        
        if not stock_col:
            return
        
        console.print(f"\n[bold cyan]Feature Quality by Stock:[/bold cyan]")
        
        # Get numeric features only
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if stock_col in numeric_cols:
            numeric_cols.remove(stock_col)
        
        if len(numeric_cols) == 0:
            console.print("[yellow]No numeric features found for quality analysis[/yellow]")
            return
        
        # Calculate feature quality metrics by stock
        stock_groups = df.groupby(stock_col)
        
        # For each stock, calculate variance and missing rate of features
        variance_by_stock = {}
        missing_by_stock = {}
        
        for stock, group in stock_groups:
            if len(group) < 10:  # Skip stocks with too few records
                continue
                
            # Calculate variance (indicator of feature informativeness)
            feature_vars = group[numeric_cols].var()
            variance_by_stock[stock] = feature_vars.mean()
            
            # Calculate missing rate
            missing_rates = group[numeric_cols].isna().mean()
            missing_by_stock[stock] = missing_rates.mean()
        
        if not variance_by_stock:
            console.print("[yellow]No stocks with sufficient data for feature quality analysis[/yellow]")
            return
        
        # Find stocks with consistently low variance (potentially problematic)
        low_variance_threshold = pd.Series(variance_by_stock).quantile(0.1)
        high_missing_threshold = 0.3
        
        problematic_stocks = []
        for stock in variance_by_stock:
            if (variance_by_stock[stock] < low_variance_threshold or 
                missing_by_stock.get(stock, 0) > high_missing_threshold):
                problematic_stocks.append({
                    'stock': stock,
                    'avg_variance': variance_by_stock[stock],
                    'avg_missing': missing_by_stock.get(stock, 0)
                })
        
        if problematic_stocks:
            console.print(f"\n[bold yellow]Stocks with potential feature quality issues:[/bold yellow]")
            for item in sorted(problematic_stocks, key=lambda x: x['avg_missing'], reverse=True)[:10]:
                console.print(f"  {item['stock']}: Variance={item['avg_variance']:.4f}, Missing={item['avg_missing']:.1%}")
        else:
            console.print("\n[green]All stocks show reasonable feature quality ✓[/green]")
    

    def _display_sample_preview(self, df: pd.DataFrame, n_samples: int = 5) -> None:
        """Display sample data preview"""
        console.print(f"\n[bold]Sample Preview ({n_samples} rows):[/bold]")
        
        # Select a subset of columns for better readability
        max_cols = 10
        if len(df.columns) > max_cols:
            sample_cols = list(df.columns[:max_cols//2]) + list(df.columns[-max_cols//2:])
            df_sample = df[sample_cols].head(n_samples)
            console.print(f"[dim]Showing {max_cols} out of {len(df.columns)} columns[/dim]")
        else:
            df_sample = df.head(n_samples)
        
        # Convert to markdown for better display
        try:
            markdown_table = df_sample.to_markdown(index=False, tablefmt="grid")
            console.print(markdown_table)
        except Exception as e:
            logger.warning(f"Failed to create markdown table: {e}")
            console.print(df_sample.to_string())
    
    def _display_data_quality_metrics(self, df: pd.DataFrame) -> None:
        """Display additional data quality metrics"""
        console.print(f"\n[bold cyan]Data Quality Metrics:[/bold cyan]")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        duplicate_ratio = duplicates / len(df)
        console.print(f"  Duplicate rows: {duplicates} ({duplicate_ratio:.2%})")
        
        if duplicate_ratio > 0.05:  # More than 5% duplicates
            console.print(f"  [bold red]⚠️  High duplicate rate detected![/bold red]")
            console.print(f"  [dim]Recommendation: Consider deduplication before training[/dim]")
        elif duplicate_ratio > 0.01:  # More than 1% duplicates
            console.print(f"  [yellow]⚠️  Moderate duplicate rate detected[/yellow]")
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            console.print(f"  Numeric columns: {len(numeric_cols)}")
            
            # Check for infinite values
            inf_counts = {}
            for col in numeric_cols:
                inf_count = df[col].isin([float('inf'), float('-inf')]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
            
            if inf_counts:
                console.print(f"  [red]Columns with infinite values: {len(inf_counts)}[/red]")
                for col, count in inf_counts.items():
                    console.print(f"    {col}: {count}")
                console.print(f"  [dim]Recommendation: Handle infinite values before training[/dim]")
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        console.print(f"  Categorical columns: {len(categorical_cols)}")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        console.print(f"  Memory usage: {memory_mb:.1f} MB")
    
    def _analyze_columns_overview(self, df_feat: pd.DataFrame, df_lab: pd.DataFrame) -> None:
        """Provide comprehensive overview of features and labels"""
        console.print(f"\n[bold cyan]Dataset Columns Overview:[/bold cyan]")
        
        # Features overview
        feat_cols = list(df_feat.columns)
        console.print(f"\n[bold green]Features Dataset ({len(feat_cols)} columns):[/bold green]")
        
        # Categorize feature columns
        numeric_feat_cols = df_feat.select_dtypes(include=['number']).columns.tolist()
        categorical_feat_cols = df_feat.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_feat_cols = df_feat.select_dtypes(include=['datetime64']).columns.tolist()
        
        console.print(f"  • Numeric features: {len(numeric_feat_cols)}")
        console.print(f"  • Categorical features: {len(categorical_feat_cols)}")
        console.print(f"  • Datetime features: {len(datetime_feat_cols)}")
        
        # Show feature column names (first few and last few)
        if len(feat_cols) > 20:
            display_cols = feat_cols[:10] + ['...'] + feat_cols[-10:]
        else:
            display_cols = feat_cols
        
        console.print(f"  • Column names: {', '.join(display_cols)}")
        
        # Labels overview
        lab_cols = list(df_lab.columns)
        console.print(f"\n[bold green]Labels Dataset ({len(lab_cols)} columns):[/bold green]")
        
        # Categorize label columns
        numeric_lab_cols = df_lab.select_dtypes(include=['number']).columns.tolist()
        categorical_lab_cols = df_lab.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_lab_cols = df_lab.select_dtypes(include=['datetime64']).columns.tolist()
        
        console.print(f"  • Numeric columns: {len(numeric_lab_cols)}")
        console.print(f"  • Categorical columns: {len(categorical_lab_cols)}")
        console.print(f"  • Datetime columns: {len(datetime_lab_cols)}")
        console.print(f"  • Column names: {', '.join(lab_cols)}")
    
    def _analyze_feature_details(self, df_feat: pd.DataFrame) -> None:
        """Detailed analysis of feature columns"""
        console.print(f"\n[bold cyan]Feature Detailed Analysis:[/bold cyan]")
        
        # Get numeric features
        numeric_cols = df_feat.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df_feat.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            # Numeric features statistics
            table = Table(title=f"Numeric Features Summary (Top {min(self.top_n, len(numeric_cols))})")
            table.add_column("Feature", style="cyan")
            table.add_column("Missing %", style="red")
            table.add_column("Mean", style="green")
            table.add_column("Std", style="yellow")
            table.add_column("Min", style="blue")
            table.add_column("Max", style="magenta")
            table.add_column("Unique", style="white")
            
            # Calculate statistics for numeric features
            stats_data = []
            for col in numeric_cols:
                missing_pct = df_feat[col].isna().mean() * 100
                mean_val = df_feat[col].mean()
                std_val = df_feat[col].std()
                min_val = df_feat[col].min()
                max_val = df_feat[col].max()
                unique_count = df_feat[col].nunique()
                
                stats_data.append({
                    'feature': col,
                    'missing_pct': missing_pct,
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'unique': unique_count
                })
            
            # Sort by missing percentage (descending)
            stats_data.sort(key=lambda x: x['missing_pct'], reverse=True)
            
            for item in stats_data[:self.top_n]:
                table.add_row(
                    item['feature'][:30],  # Truncate long names
                    f"{item['missing_pct']:.1f}%",
                    f"{item['mean']:.4f}" if pd.notna(item['mean']) else "N/A",
                    f"{item['std']:.4f}" if pd.notna(item['std']) else "N/A",
                    f"{item['min']:.4f}" if pd.notna(item['min']) else "N/A",
                    f"{item['max']:.4f}" if pd.notna(item['max']) else "N/A",
                    str(item['unique'])
                )
            
            console.print("\n")
            console.print(table)
        
        if categorical_cols:
            # Categorical features analysis
            table = Table(title=f"Categorical Features Summary")
            table.add_column("Feature", style="cyan")
            table.add_column("Missing %", style="red")
            table.add_column("Unique Count", style="green")
            table.add_column("Most Frequent", style="yellow")
            table.add_column("Freq Count", style="blue")
            
            for col in categorical_cols[:self.top_n]:
                missing_pct = df_feat[col].isna().mean() * 100
                unique_count = df_feat[col].nunique()
                
                # Get most frequent value
                value_counts = df_feat[col].value_counts()
                if len(value_counts) > 0:
                    most_freq_val = str(value_counts.index[0])[:20]  # Truncate
                    most_freq_count = value_counts.iloc[0]
                else:
                    most_freq_val = "N/A"
                    most_freq_count = 0
                
                table.add_row(
                    col[:30],
                    f"{missing_pct:.1f}%",
                    str(unique_count),
                    most_freq_val,
                    str(most_freq_count)
                )
            
            console.print("\n")
            console.print(table)
        
        # Feature quality insights
        console.print(f"\n[bold yellow]Feature Quality Insights:[/bold yellow]")
        
        # Zero variance features
        zero_var_features = []
        low_var_features = []
        
        for col in numeric_cols:
            var = df_feat[col].var()
            if pd.notna(var):
                if var == 0:
                    zero_var_features.append(col)
                elif var < 1e-10:
                    low_var_features.append(col)
        
        if zero_var_features:
            console.print(f"  • Zero variance features ({len(zero_var_features)}): {', '.join(zero_var_features[:5])}{'...' if len(zero_var_features) > 5 else ''}")
        
        if low_var_features:
            console.print(f"  • Very low variance features ({len(low_var_features)}): {', '.join(low_var_features[:5])}{'...' if len(low_var_features) > 5 else ''}")
        
        # High cardinality categorical features
        high_cardinality_features = [col for col in categorical_cols if df_feat[col].nunique() > len(df_feat) * 0.8]
        if high_cardinality_features:
            console.print(f"  • High cardinality categorical features: {', '.join(high_cardinality_features)}")
        
        # Constant features
        constant_features = [col for col in df_feat.columns if df_feat[col].nunique() <= 1]
        if constant_features:
            console.print(f"  • Constant features: {', '.join(constant_features)}")
    
    def _analyze_label_details(self, df_lab: pd.DataFrame) -> None:
        """Detailed analysis of label columns"""
        console.print(f"\n[bold cyan]Label Detailed Analysis:[/bold cyan]")
        
        # Create detailed table for all label columns
        table = Table(title="Label Columns Analysis")
        table.add_column("Label Column", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Missing %", style="red")
        table.add_column("Unique Values", style="yellow")
        table.add_column("Distribution", style="blue")
        
        for col in df_lab.columns:
            missing_pct = df_lab[col].isna().mean() * 100
            unique_count = df_lab[col].nunique()
            col_type = str(df_lab[col].dtype)
            
            # Get value distribution
            value_counts = df_lab[col].value_counts()
            if len(value_counts) <= 5:
                distribution = ", ".join([f"{k}:{v}" for k, v in value_counts.head().items()])
            else:
                top_3 = ", ".join([f"{k}:{v}" for k, v in value_counts.head(3).items()])
                distribution = f"{top_3}..."
            
            table.add_row(
                col[:25],
                col_type,
                f"{missing_pct:.1f}%",
                str(unique_count),
                distribution[:40]  # Truncate long distributions
            )
        
        console.print("\n")
        console.print(table)
        
        # Detailed analysis for each label column
        for col in df_lab.columns:
            console.print(f"\n[bold yellow]Label Column: {col}[/bold yellow]")
            
            # Basic statistics
            total_count = len(df_lab)
            null_count = df_lab[col].isna().sum()
            valid_count = total_count - null_count
            
            console.print(f"  • Total records: {total_count:,}")
            console.print(f"  • Valid records: {valid_count:,}")
            console.print(f"  • Missing records: {null_count:,} ({null_count/total_count:.2%})")
            
            # Value distribution
            value_counts = df_lab[col].value_counts(dropna=False)
            console.print(f"  • Unique values: {len(value_counts)}")
            
            # Show distribution
            console.print(f"  • Value distribution:")
            for value, count in value_counts.head(10).items():
                percentage = count / total_count
                console.print(f"    {value}: {count:,} ({percentage:.2%})")
            
            if len(value_counts) > 10:
                console.print(f"    ... and {len(value_counts) - 10} more values")
            
            # For numeric labels, show statistical summary
            if pd.api.types.is_numeric_dtype(df_lab[col]):
                numeric_stats = df_lab[col].describe()
                console.print(f"  • Statistical summary:")
                console.print(f"    Mean: {numeric_stats['mean']:.4f}")
                console.print(f"    Std: {numeric_stats['std']:.4f}")
                console.print(f"    Min: {numeric_stats['min']:.4f}")
                console.print(f"    Max: {numeric_stats['max']:.4f}")
                console.print(f"    25%: {numeric_stats['25%']:.4f}")
                console.print(f"    50%: {numeric_stats['50%']:.4f}")
                console.print(f"    75%: {numeric_stats['75%']:.4f}")
    
    def _analyze_missing_patterns(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> None:
        """Analyze missing value patterns in detail"""
        console.print(f"\n[bold cyan]{dataset_name} Missing Value Patterns:[/bold cyan]")
        
        # Overall missing statistics
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isna().sum().sum()
        overall_missing_rate = total_missing / total_cells
        
        console.print(f"  • Total cells: {total_cells:,}")
        console.print(f"  • Missing cells: {total_missing:,}")
        console.print(f"  • Overall missing rate: {overall_missing_rate:.2%}")
        
        # Missing by column
        missing_by_col = df.isna().sum().sort_values(ascending=False)
        missing_cols = missing_by_col[missing_by_col > 0]
        
        if len(missing_cols) > 0:
            table = Table(title=f"Missing Values by Column (Top {min(self.top_n, len(missing_cols))})")
            table.add_column("Column", style="cyan")
            table.add_column("Missing Count", style="red")
            table.add_column("Missing %", style="yellow")
            table.add_column("Data Type", style="green")
            
            for col, missing_count in missing_cols.head(self.top_n).items():
                missing_pct = missing_count / len(df) * 100
                dtype = str(df[col].dtype)
                
                table.add_row(
                    col[:30],
                    f"{missing_count:,}",
                    f"{missing_pct:.1f}%",
                    dtype
                )
            
            console.print("\n")
            console.print(table)
            
            # Missing value categories
            complete_cols = len(df.columns) - len(missing_cols)
            high_missing_cols = (missing_by_col > len(df) * 0.5).sum()
            medium_missing_cols = ((missing_by_col > len(df) * 0.1) & (missing_by_col <= len(df) * 0.5)).sum()
            low_missing_cols = ((missing_by_col > 0) & (missing_by_col <= len(df) * 0.1)).sum()
            
            console.print(f"\n[bold yellow]Missing Value Categories:[/bold yellow]")
            console.print(f"  • Complete columns (0% missing): {complete_cols}")
            console.print(f"  • Low missing (0-10%): {low_missing_cols}")
            console.print(f"  • Medium missing (10-50%): {medium_missing_cols}")
            console.print(f"  • High missing (>50%): {high_missing_cols}")
            
            if high_missing_cols > 0:
                console.print(f"\n[bold red]⚠️  Columns with >50% missing data:[/bold red]")
                high_missing_col_names = missing_by_col[missing_by_col > len(df) * 0.5].index.tolist()
                for col in high_missing_col_names[:5]:
                    missing_pct = missing_by_col[col] / len(df) * 100
                    console.print(f"    {col}: {missing_pct:.1f}% missing")
                if len(high_missing_col_names) > 5:
                    console.print(f"    ... and {len(high_missing_col_names) - 5} more")
        else:
            console.print("\n[green]✓ No missing values found in any column![/green]")
    
    def run_diagnosis(self) -> None:
        """Run complete data diagnosis"""
        start_time = time.time()
        
        try:
            console.print(f"[bold green]Starting diagnosis for: {self.root}[/bold green]")
            console.print(f"[dim]Pattern: {self.month_pattern}[/dim]\n")
            
            # Find files
            feat_files, lab_files = self._find_parquet_files()
            
            # Load data
            df_feat = self._load_parquet_files(feat_files, "features")
            df_lab = self._load_parquet_files(lab_files, "labels")
            
            # Display basic info
            console.print(f"\n[bold cyan]Dataset Overview:[/bold cyan]")
            console.print(f"  Feature dataset shape: {df_feat.shape}")
            console.print(f"  Label dataset shape: {df_lab.shape}")
            
            # Merge data
            df_merged = self._merge_data(df_feat, df_lab)
            console.print(f"  Merged dataset shape: {df_merged.shape}")
            
            # Run comprehensive analyses
            # 1. Dataset overview
            self._analyze_columns_overview(df_feat, df_lab)
            
            # 2. Feature analysis
            self._analyze_feature_details(df_feat)
            self._analyze_missing_patterns(df_feat, "Features")
            
            # 3. Label analysis
            self._analyze_label_details(df_lab)
            self._analyze_missing_patterns(df_lab, "Labels")
            
            # 4. Merged data analysis
            self._analyze_labels(df_merged)
            self._display_data_quality_metrics(df_merged)
            self._analyze_missing_patterns(df_merged, "Merged Data")
            
            # 5. Stock-level analysis
            self._analyze_by_stock(df_merged)
            self._analyze_stock_feature_quality(df_merged)
            
            # 6. Sample preview
            self._display_sample_preview(df_merged)
            
            # Performance info
            elapsed_time = time.time() - start_time
            console.print(f"\n[bold green]✓ Diagnosis completed in {elapsed_time:.2f} seconds[/bold green]")
            
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            console.print(f"[bold red]❌ Diagnosis failed: {e}[/bold red]")
            raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Feature/Label quick diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnose_feat_label.py --root "/obs/users/fenglang/general/Spoofing Detect/data" --month 202504
  python diagnose_feat_label.py --root /path/to/data --topn 30 --samples 10
  python diagnose_feat_label.py --root /path/to/data --month 202504 --save-stock-analysis
        """
    )
    parser.add_argument("--root", required=True, 
                       help="Parent dir containing features_select & labels_select")
    parser.add_argument("--month", default="", 
                       help="Glob pattern like 202504 (optional)")
    parser.add_argument("--topn", type=int, default=20, 
                       help="Number of features to show for NA rate ranking")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of sample rows to display")
    parser.add_argument("--save-stock-analysis", action="store_true",
                       help="Save detailed stock-level analysis to CSV")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        diagnostics = DataDiagnostics(
            root_path=args.root,
            month_pattern=args.month,
            top_n=args.topn,
            save_stock_analysis=args.save_stock_analysis
        )
        diagnostics.run_diagnosis()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Diagnosis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


"""
python scripts/analysis/diagnose_feat_label.py --root "/obs/users/fenglang/general/Spoofing Detect/data" --month 202504 --topn 20
python scripts/analysis/diagnose_feat_label.py --root "/obs/users/fenglang/general/Spoofing Detect/data" --month 202504 --topn 20 --save-stock-analysis
"""