# src/prep_and_train_augmented.py (with MLflow)
import pandas as pd, numpy as np, joblib, json, datetime, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from .synth_features import simulate_weather, simulate_traffic, holiday_flag

# MLflow (optional) — tracks locally to ./mlruns by default
import mlflow
from mlflow import log_metric, log_param, log_artifacts, set_tracking_uri

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "historical_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "artifacts_aug"
FEATURES_CSV = PROCESSED_DIR / "features_aug.csv"
MODEL_PATH = OUT_DIR / "rf_model_aug.joblib"
FEATURE_COLS_JSON = OUT_DIR / "feature_columns_aug.json"
CONFORMAL_Q90 = OUT_DIR / "conformal_q90.npy"
META_JSON = OUT_DIR / "model_metadata.json"

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

def main():
    # Set MLflow store (local folder)
    set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print(f"CSV not found at {CSV_PATH}. Place your file there.")
        return

    df = pd.read_csv(CSV_PATH)
    try:
        from utils.validate import basic_check
        missing = basic_check(df)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    except Exception as e:
        print(f"[validate] {e if str(e) else 'skipped'}")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"], errors="coerce")
    df = df.dropna(subset=["created_at", "actual_delivery_time"]).copy()

    num_cols = [
        "market_id","order_protocol","total_items","subtotal","num_distinct_items",
        "min_item_price","max_item_price","total_onshift_dashers","total_busy_dashers",
        "total_outstanding_orders","estimated_order_place_duration",
        "estimated_store_to_consumer_driving_duration"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().copy()

    df["delivery_seconds"] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds()
    df = df[(df["delivery_seconds"] >= 5*60) & (df["delivery_seconds"] <= 3*3600)].copy()

    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["is_holiday"] = holiday_flag(df["created_at"])

    def meal_period(h):
        if 6 <= h <= 10:  return "breakfast"
        if 11 <= h <= 14: return "lunch"
        if 18 <= h <= 21: return "dinner"
        return "other"
    df["meal_period"] = df["hour"].apply(meal_period)
    df["peak_hour"] = df["hour"].isin([7,8,9,12,13,18,19]).astype(int)
    df["is_rush"] = ((df["hour"].between(7,9)) | (df["hour"].between(16,19))).astype(int)

    wdf = simulate_weather(df["created_at"])
    df = pd.concat([df.reset_index(drop=True), wdf.reset_index(drop=True)], axis=1)
    df["traffic"] = simulate_traffic(df["hour"], df["day_of_week"], df["is_holiday"])

    df["load_ratio"]      = df["total_outstanding_orders"] / (df["total_onshift_dashers"] + 1e-3)
    df["busy_ratio"]      = df["total_busy_dashers"]      / (df["total_onshift_dashers"] + 1e-3)
    df["demand_pressure"] = df["total_outstanding_orders"] / (df["total_busy_dashers"] + 1e-3)
    df["subtotal_per_item"] = df["subtotal"] / (df["total_items"] + 1e-3)
    df["price_range"] = df["max_item_price"] - df["min_item_price"]

    cat_cols = []
    if "store_primary_category" in df.columns:
        cat_cols.append("store_primary_category")
    df["order_protocol"] = df["order_protocol"].astype("category")
    cat_cols.append("order_protocol")
    cat_cols += ["weather","traffic","meal_period"]

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
    X_cat = enc.fit_transform(df[cat_cols])
    names = enc.get_feature_names_out(cat_cols)
    df_cat = pd.DataFrame(X_cat, columns=names, index=df.index)
    df = pd.concat([df.drop(columns=cat_cols), df_cat], axis=1)

    feature_cols = [
        "total_items","num_distinct_items","subtotal","min_item_price","max_item_price",
        "subtotal_per_item","price_range",
        "total_onshift_dashers","total_busy_dashers","total_outstanding_orders",
        "load_ratio","busy_ratio","demand_pressure",
        "hour","day_of_week","is_weekend","is_holiday","peak_hour","is_rush",
        "estimated_order_place_duration","estimated_store_to_consumer_driving_duration",
        "market_id","temperature_c","precip"
    ]
    feature_cols += [c for c in df.columns if c.startswith((
        "store_primary_category_","order_protocol_","weather_","traffic_","meal_period_"))]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_model = df[feature_cols + ["delivery_seconds"]].copy()
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(FEATURES_CSV, index=False)

    X = df_model[feature_cols]; y = df_model["delivery_seconds"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="rf_augmented"):
        model = RandomForestRegressor(
            n_estimators=200, max_depth=18, min_samples_leaf=3, min_samples_split=6,
            n_jobs=-1, random_state=42
        ).fit(X_train, y_train)

        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(((y_test - pred) ** 2).mean()))
        print(f"[AUGMENTED] MAE: {mae:.1f} sec | RMSE: {rmse:.1f} sec")

        # Conformal
        residuals = np.abs(y_test - pred)
        q90 = float(np.quantile(residuals, 0.90))
        np.save(CONFORMAL_Q90, q90)

        # Save artifacts
        joblib.dump(model, MODEL_PATH, compress=3)
        FEATURE_COLS_JSON.write_text(json.dumps(list(X.columns)))
        meta = {
            "model_type": "RandomForestRegressor",
            "train_datetime_utc": datetime.datetime.utcnow().isoformat(),
            "n_samples": int(len(df_model)),
            "n_features": int(len(feature_cols)),
            "mae_sec": float(mae),
            "rmse_sec": float(rmse),
            "conformal_q90_sec": q90
        }
        save_json(meta, META_JSON)

        # MLflow logging
        log_param("model_type", "RandomForestRegressor")
        log_param("n_estimators", 200)
        log_param("max_depth", 18)
        log_param("min_samples_leaf", 3)
        log_param("min_samples_split", 6)
        log_metric("mae_sec", float(mae))
        log_metric("rmse_sec", float(rmse))
        log_metric("conformal_q90_sec", q90)
        # log artifacts folder (contains model, feature cols, metadata, q90)
        log_artifacts(str(OUT_DIR))

    print(f"Saved → {MODEL_PATH}")
    print(f"Saved → {FEATURE_COLS_JSON}")
    print(f"Saved → {CONFORMAL_Q90}")
    print(f"Saved → {META_JSON}")
    print(f"Saved → {FEATURES_CSV}  shape={df_model.shape}")

if __name__ == "__main__":
    main()
