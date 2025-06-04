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
    
    def __init__(self, root_path: str, month_pattern: str = "", top_n: int = 20):
        self.root = Path(root_path)
        self.month_pattern = f"*{month_pattern}*" if month_pattern else "*"
        self.top_n = top_n
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
        
        table = Table(title=f"Label Distribution ({label_col})")
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
    
    def _analyze_missing_values(self, df_feat: pd.DataFrame) -> None:
        """Analyze and display missing value statistics"""
        na_rates = df_feat.isna().mean().sort_values(ascending=False)
        
        if na_rates.empty:
            console.print("[green]No missing values found![/green]")
            return
        
        top_na_rates = na_rates.head(self.top_n)
        
        table = Table(title=f"Top {self.top_n} Features by Missing Rate")
        table.add_column("Feature", style="cyan")
        table.add_column("Missing Rate", style="red")
        table.add_column("Missing Count", style="yellow")
        
        total_rows = len(df_feat)
        for feature, rate in top_na_rates.items():
            if rate > 0:  # Only show features with missing values
                missing_count = int(rate * total_rows)
                table.add_row(feature[:50], f"{rate:.2%}", str(missing_count))
        
        console.print("\n")
        console.print(table)
        
        # Summary statistics
        high_missing = (na_rates > 0.5).sum()
        medium_missing = ((na_rates > 0.1) & (na_rates <= 0.5)).sum()
        
        console.print(f"\n[bold yellow]Missing Value Summary:[/bold yellow]")
        console.print(f"  Features with >50% missing: {high_missing}")
        console.print(f"  Features with 10-50% missing: {medium_missing}")
        console.print(f"  Features with <10% missing: {len(na_rates) - high_missing - medium_missing}")
    
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
            
            # Run analyses
            self._analyze_labels(df_merged)
            self._analyze_missing_values(df_feat)
            self._display_data_quality_metrics(df_merged)
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
            top_n=args.topn
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
python scripts/data_process/diagnose_feat_label.py --root "/obs/users/fenglang/general/Spoofing Detect/data" --month 202504 --topn 20
"""