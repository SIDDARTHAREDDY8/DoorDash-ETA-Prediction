# DoorDash ETA Predictor — Full Code (No Dataset)

This archive contains the full **code** (base + upgrades).  
Add your dataset at `data/raw/historical_data.csv` and follow the steps.

## Quickstart
```bash
# 1) Create & activate a virtual env
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Place your CSV
#   expected filename: data/raw/historical_data.csv
#   required columns (from the DoorDash public schema):
#   created_at, actual_delivery_time, market_id, store_primary_category, order_protocol,
#   total_items, subtotal, num_distinct_items, min_item_price, max_item_price,
#   total_onshift_dashers, total_busy_dashers, total_outstanding_orders,
#   estimated_order_place_duration, estimated_store_to_consumer_driving_duration

# 4) Train baseline (augmented) model
export MPLBACKEND=Agg   # avoids macOS OpenGL warnings
python -m src.prep_and_train_augmented

# 5) Optional: train advanced models (XGBoost + Optuna + ensemble)
python -m src.train_advanced

# 6) Optional: SHAP explainability (saves artifacts_aug/shap_summary.png)
export MPLBACKEND=Agg
python -m utils.explain_shap

# 7) Run the upgraded Streamlit app
streamlit run app/app_pro.py --server.headless true
```

## Extras
- **Batch predictions:** `python -m utils.batch_predict data/processed/features_aug.csv data/processed/pred_features_aug.csv --model xgb`
- **Scheduler (simulated real-time):** `python -m scripts.schedule_predict` then drop CSVs into `data/processed/incoming/`
- **Weather/Traffic API hooks:** set `OPENWEATHER_API_KEY` and `TOMTOM_API_KEY`, then call functions in `src/apis.py` during your own augmentation step.