# utils/explain_shap.py
import shap, joblib, json, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts_aug"
FEATURES = PROJECT_ROOT / "data" / "processed" / "features_aug.csv"

def main():
    # Prefer XGB if present, else RF
    model_path = ART / "xgb_model_aug.joblib"
    if not model_path.exists():
        model_path = ART / "rf_model_aug.joblib"
    if not model_path.exists():
        print("No model found. Train with: python -m src.prep_and_train_augmented or python -m src.train_advanced")
        return

    model = joblib.load(model_path)
    feature_cols = json.loads((ART / "feature_columns_aug.json").read_text())
    df = pd.read_csv(FEATURES)
    X = df[feature_cols].sample(n=min(2000, len(df)), random_state=42)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    plt.tight_layout()
    shap.summary_plot(shap_values, X, show=False)
    (ART / "shap_summary.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ART / "shap_summary.png", dpi=150, bbox_inches="tight")
    print("Saved SHAP summary to artifacts_aug/shap_summary.png")

if __name__ == "__main__":
    main()
