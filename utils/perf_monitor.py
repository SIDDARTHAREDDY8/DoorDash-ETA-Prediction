from pathlib import Path
import pandas as pd, json, joblib

PROJECT = Path(__file__).resolve().parents[1]
ART = PROJECT / "artifacts_aug"
FEATS = PROJECT / "data" / "processed" / "features_aug.csv"
REPORTS = PROJECT / "reports"

def rolling_mae(window: int = 1000) -> Path:
    """
    Computes a rolling MAE over predictions on the full processed dataset.
    Writes reports/rolling_mae_w{window}.csv with a single 'rolling_mae' column.
    """
    df = pd.read_csv(FEATS)
    feature_cols = json.loads((ART / "feature_columns_aug.json").read_text())
    if (ART / "xgb_model_aug.joblib").exists():
        model = joblib.load(ART / "xgb_model_aug.joblib")
    else:
        model = joblib.load(ART / "rf_model_aug.joblib")
    df["pred"] = model.predict(df[feature_cols])
    err = (df["delivery_seconds"] - df["pred"]).abs()
    roll = err.rolling(window=window, min_periods=max(100, window // 10)).mean()
    out = REPORTS / f"rolling_mae_w{window}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    roll.to_csv(out, header=["rolling_mae"], index=False)
    return out

if __name__ == "__main__":
    p = rolling_mae(1000)
    print("Saved:", p)

