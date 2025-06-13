 python scripts/data_process/raw_data/merge_event_stream.py     --root "/home/ma-user/code/fenglang/Spoofing Detect/data/base_data"     --tickers 000989.SZ 300233.SZ

 python scripts/data_process/features/feature_generator.py     --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream"     --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/features"     --backend polars     --extended

 python scripts/data_process/labels/label_generator.py     --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream"     --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/labels_enhanced"     --r1_ms 1000 --r2_ms 1000 --r2_mult 1.0     --extended     --backend pandas


 python scripts/train/train.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_ensemble \
  --eval_output_dir "results/train_results"