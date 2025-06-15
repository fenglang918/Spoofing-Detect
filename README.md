python scripts/data_process/raw_data/merge_event_stream.py \
    --root "/home/ma-user/code/fenglang/Spoofing Detect/data/base_data" \
    --tickers 300233.SZ \
    --filter-hours


python scripts/data_process/labels/label_generator.py \
    --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/labels_enhanced" \
    --r1_ms 1000 --r2_ms 1000 --r2_mult 1.0 \
    --tickers 300233.SZ \
    --extended \
    --backend polars


python scripts/data_process/features/feature_generator.py \
    --input_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/event_stream" \
    --output_dir "/home/ma-user/code/fenglang/Spoofing Detect/data/features" \
    --tickers 300233.SZ \
    --backend polars \
    --extended


python scripts/train/train.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --train_regex "202503|202504" \
  --valid_regex "202505" \
  --sampling_method "none" \
  --use_ensemble \
  --eval_output_dir "results/train_results"

python scripts/analysis/model_prediction_visualization.py \
  --data_root "/home/ma-user/code/fenglang/Spoofing Detect/data" \
  --model_path "results/trained_models/spoofing_model_Enhanced_none_Ensemble.pkl" \
  --valid_regex "202505" \
  --output_dir "results/prediction_visualization/202505_full_month" \
  --prob_threshold 0.01 \
  --top_k_percent 0.005 \
  --max_plots 50