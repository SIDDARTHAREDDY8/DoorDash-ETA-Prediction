# src/train_advanced.py (with MLflow)
import pandas as pd, joblib, json, numpy as np, datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import mlflow
from mlflow import log_metric, log_param, log_artifacts, set_tracking_uri

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features_aug.csv"
ART = PROJECT_ROOT / "artifacts_aug"
MODEL_PATH = ART / "xgb_model_aug.joblib"
FEATURE_COLS_JSON = ART / "feature_columns_aug.json"
META_JSON = ART / "model_metadata.json"

def main():
    set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    y = df["delivery_seconds"]
    X = df.drop(columns=["delivery_seconds"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="xgb_advanced"):
        model = XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(((y_test - pred) ** 2).mean()))
        print(f"[XGB] MAE: {mae:.1f} sec | RMSE: {rmse:.1f} sec")

        joblib.dump(model, MODEL_PATH, compress=3)
        FEATURE_COLS_JSON.write_text(json.dumps(list(X.columns)))
        meta = {
            "model_type": "XGBRegressor",
            "train_datetime_utc": datetime.datetime.utcnow().isoformat(),
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
            "mae_sec": float(mae),
            "rmse_sec": float(rmse)
        }
        META_JSON.write_text(json.dumps(meta, indent=2))

        # MLflow logging
        log_param("model_type", "XGBRegressor")
        log_param("n_estimators", 300)
        log_param("max_depth", 8)
        log_param("learning_rate", 0.1)
        log_param("subsample", 0.8)
        log_param("colsample_bytree", 0.8)
        log_metric("mae_sec", float(mae))
        log_metric("rmse_sec", float(rmse))
        log_artifacts(str(ART))

    print(f"Saved → {MODEL_PATH}")
    print(f"Saved → {FEATURE_COLS_JSON}")
    print(f"Saved → {META_JSON}")

if __name__ == "__main__":
    main()
