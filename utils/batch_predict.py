import json, joblib, pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts_aug"

def main(in_csv: str, out_csv: str, model: str="xgb"):
    feature_cols = json.loads((ART / "feature_columns_aug.json").read_text())
    df = pd.read_csv(in_csv)
    X = df[feature_cols]
    model_path = ART / ("xgb_model_aug.joblib" if model == "xgb" else "rf_model_aug.joblib")
    mdl = joblib.load(model_path)
    df["pred_eta_seconds"] = mdl.predict(X)
    df.to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv"); ap.add_argument("out_csv")
    ap.add_argument("--model", choices=["xgb","rf"], default="xgb")
    args = ap.parse_args()
    main(args.in_csv, args.out_csv, args.model)