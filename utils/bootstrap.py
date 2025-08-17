# utils/bootstrap.py
from pathlib import Path
from typing import Dict
import json, joblib, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

PROJECT = Path(__file__).resolve().parents[1]
ART = PROJECT / "artifacts_aug"
DATA = PROJECT / "data"
RAW = DATA / "raw" / "historical_data.csv"
PROC = DATA / "processed" / "features_aug.csv"
FEAT_JSON = ART / "feature_columns_aug.json"
MODEL = ART / "rf_model_aug.joblib"
META = ART / "model_metadata.json"

def _basic_featurize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    # timestamps & target
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"], errors="coerce")
    df = df.dropna(subset=["created_at","actual_delivery_time"])
    df["delivery_seconds"] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds()
    df = df[(df["delivery_seconds"] >= 5*60) & (df["delivery_seconds"] <= 3*3600)]

    # numerics
    for c in [
        "total_items","num_distinct_items","subtotal","min_item_price","max_item_price",
        "total_onshift_dashers","total_busy_dashers","total_outstanding_orders",
        "estimated_order_place_duration","estimated_store_to_consumer_driving_duration","market_id"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # simple features
    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    eps = 1e-3
    df["load_ratio"] = df["total_outstanding_orders"] / (df["total_onshift_dashers"] + eps)
    df["busy_ratio"] = df["total_busy_dashers"] / (df["total_onshift_dashers"] + eps)
    df["demand_pressure"] = df["total_outstanding_orders"] / (df["total_busy_dashers"] + eps)
    df["subtotal_per_item"] = df["subtotal"] / (df["total_items"] + eps)
    df["price_range"] = df["max_item_price"] - df["min_item_price"]
    df["peak_hour"] = df["hour"].isin([7,8,9,12,13,18,19]).astype(int)
    df["is_rush"] = ((df["hour"].between(7,9)) | (df["hour"].between(16,19))).astype(int)

    feature_cols = [
        "total_items","num_distinct_items","subtotal","min_item_price","max_item_price",
        "subtotal_per_item","price_range",
        "total_onshift_dashers","total_busy_dashers","total_outstanding_orders",
        "load_ratio","busy_ratio","demand_pressure",
        "hour","day_of_week","is_weekend","peak_hour","is_rush",
        "estimated_order_place_duration","estimated_store_to_consumer_driving_duration",
        "market_id"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=feature_cols + ["delivery_seconds"])
    return df[feature_cols + ["delivery_seconds"]], feature_cols

def ensure_small_model(max_rows: int = 8000) -> Dict:
    """
    Ensures rf_model_aug.joblib + feature_columns_aug.json exist.
    Uses processed CSV if present; otherwise builds from raw CSV.
    """
    ART.mkdir(parents=True, exist_ok=True)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)

    if MODEL.exists() and FEAT_JSON.exists():
        return {"created": False, "reason": "artifacts already exist"}

    # Pick source
    df = None
    feature_cols = None
    if PROC.exists():
        try:
            df = pd.read_csv(PROC)
            if "delivery_seconds" in df.columns:
                feature_cols = [c for c in df.columns if c != "delivery_seconds"]
        except Exception:
            df = None

    if df is None:
        if not RAW.exists():
            raise FileNotFoundError(
                "No artifacts and no data found. Commit a small data/processed/features_aug.csv "
                "or data/raw/historical_data.csv to the repo."
            )
        raw = pd.read_csv(RAW)
        df, feature_cols = _basic_featurize(raw)
        # keep a small processed file to speed future boots
        df.sample(n=min(len(df), 15000), random_state=42).to_csv(PROC, index=False)

    # Subsample to keep memory small on free tiers
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    X = df[feature_cols]; y = df["delivery_seconds"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=120, max_depth=16, min_samples_leaf=3, min_samples_split=6,
        n_jobs=-1, random_state=42
    ).fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    rmse = float(np.sqrt(((yte - pred) ** 2).mean()))

    joblib.dump(model, MODEL, compress=3)
    FEAT_JSON.write_text(json.dumps(feature_cols))
    META.write_text(json.dumps({
        "train_datetime_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mae_sec": mae,
        "rmse_sec": rmse,
        "n_features": int(len(feature_cols)),
        "n_samples": int(len(df)),
        "note": "Auto-trained compact RF on first run (bootstrap)."
    }, indent=2))

    return {"created": True, "mae": mae, "rmse": rmse, "n": int(len(df))}

