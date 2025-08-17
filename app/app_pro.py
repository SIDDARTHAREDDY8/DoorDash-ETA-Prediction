# app/app_pro.py — DoorDash ETA Prediction Dashboard (single page)
import sys
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px

# ── Make project root importable so `utils/` works when launched via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config + theme
st.set_page_config(page_title="DoorDash ETA Prediction Dashboard", page_icon="🛵", layout="wide")

# ── Paths & artifact loaders
ART = PROJECT_ROOT / "artifacts_aug"
DATA = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

def load_logo_bytes():
    logo_path = PROJECT_ROOT / "app" / "assets" / "logo.png"
    return logo_path.read_bytes() if logo_path.exists() else None

@st.cache_resource
def load_artifacts():
    feature_cols = []
    if (ART / "feature_columns_aug.json").exists():
        feature_cols = json.loads((ART / "feature_columns_aug.json").read_text())
    rf = joblib.load(ART / "rf_model_aug.joblib") if (ART / "rf_model_aug.joblib").exists() else None
    xgb = joblib.load(ART / "xgb_model_aug.joblib") if (ART / "xgb_model_aug.joblib").exists() else None
    meta = json.loads((ART / "model_metadata.json").read_text()) if (ART / "model_metadata.json").exists() else {}
    q90 = None
    if (ART / "conformal_q90.npy").exists():
        try:
            q90 = float(np.load(ART / "conformal_q90.npy"))
        except Exception:
            q90 = None
    return feature_cols, rf, xgb, meta, q90

feature_cols, rf_model, xgb_model, meta, q90 = load_artifacts()

# ── Sidebar: title, logo, snapshot
with st.sidebar:
    st.markdown("### 🛵 ETA Dashboard")
    lb = load_logo_bytes()
    if lb:
        try:
            st.image(BytesIO(lb), use_container_width=True)
        except TypeError:
            st.image(BytesIO(lb), use_column_width=True)  # fallback for older Streamlit
    else:
        st.markdown("**📦 DoorDash ETA**")
    st.caption("Predict delivery ETAs with real-world signals (weather, traffic, supply/demand).")

with st.sidebar:
    st.divider()
    st.markdown("**Model Snapshot**")
    mdl = xgb_model or rf_model
    st.write(f"Type: **{'XGBoost' if xgb_model else 'RandomForest'}**" if mdl else "No model loaded")
    if meta:
        if isinstance(meta.get("mae_sec", None), (float, int)):
            st.write(f"Trained (UTC): `{meta.get('train_datetime_utc','?')}`")
            st.write(f"MAE: **{meta.get('mae_sec'):.1f} s**  |  RMSE: **{meta.get('rmse_sec'):.1f} s**")
        st.write(f"Features: **{meta.get('n_features','?')}**  |  Rows: **{meta.get('n_samples','?')}**")
    if mdl and feature_cols and hasattr(mdl, "feature_importances_"):
        imp = np.asarray(mdl.feature_importances_)
        top = imp.argsort()[-10:][::-1]
        st.write("Top features:")
        for i in top:
            st.write(f"• {feature_cols[i]} — {imp[i]:.3f}")

# ── Title & Tabs (named to avoid index errors)
st.title("📦 DoorDash ETA Prediction Dashboard")
TAB_NAMES = ["Single Prediction","Batch Upload","Trends","Scenario Simulator","Explainability","Monitoring"]
tabs = st.tabs(TAB_NAMES)
tab = dict(zip(TAB_NAMES, tabs))

# ── Tab: Single Prediction
with tab["Single Prediction"]:
    st.subheader("Single Prediction")
    preset = st.selectbox("Preset", ["Custom","Lunch rush","Dinner peak","Rainy weekend","Late night"])

    # defaults
    total_items, num_distinct_items = 3, 2
    subtotal, min_item_price, max_item_price = 3000.0, 800.0, 1800.0
    onshift, busy, outstanding = 25, 18, 30
    hour, dow = 19, 5
    is_holiday = False
    temperature_c, precip = 20.0, False
    est_place, est_drive = 300, 900

    if preset != "Custom":
        if preset == "Lunch rush":
            hour, dow, busy, outstanding, temperature_c, precip = 12, 2, 40, 120, 23.0, False
        elif preset == "Dinner peak":
            hour, dow, busy, outstanding, temperature_c, precip = 19, 5, 60, 200, 20.0, False
        elif preset == "Rainy weekend":
            hour, dow, busy, outstanding, temperature_c, precip = 18, 6, 50, 180, 12.0, True
        elif preset == "Late night":
            hour, dow, busy, outstanding, temperature_c, precip = 23, 5, 10, 30, 18.0, False

    c1,c2,c3 = st.columns(3)
    with c1:
        total_items = st.number_input("Total Items", 1, 50, total_items)
        num_distinct_items = st.number_input("Distinct Items", 1, 50, num_distinct_items)
        subtotal = st.number_input("Subtotal", 0.0, 100000.0, subtotal, 50.0)
        min_item_price = st.number_input("Min Item Price", 0.0, 100000.0, min_item_price, 50.0)
        max_item_price = st.number_input("Max Item Price", 0.0, 100000.0, max_item_price, 50.0)
    with c2:
        onshift = st.number_input("Onshift Dashers", 0, 500, onshift)
        busy = st.number_input("Busy Dashers", 0, 500, busy)
        outstanding = st.number_input("Outstanding Orders", 0, 2000, outstanding)
        hour = st.slider("Hour", 0, 23, hour)
        dow = st.slider("Day of Week (Mon=0)", 0, 6, dow)
    with c3:
        is_holiday = st.checkbox("Holiday", is_holiday)
        temperature_c = st.number_input("Temperature (°C)", -20.0, 45.0, temperature_c, 0.5)
        precip = st.checkbox("Precipitation", precip)
        est_place = st.number_input("Est. Order Place (sec)", 0, 7200, est_place)
        est_drive = st.number_input("Est. Drive (sec)", 0, 7200, est_drive)

    # derived
    load_ratio = outstanding/(onshift+1e-3)
    busy_ratio = busy/(onshift+1e-3)
    demand_pressure = outstanding/(busy+1e-3)
    subtotal_per_item = subtotal/(total_items+1e-3)
    price_range = max_item_price - min_item_price
    is_weekend = 1 if dow in [5,6] else 0
    peak_hour = 1 if hour in [7,8,9,12,13,18,19] else 0
    is_rush = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0

    row = {
        "total_items": total_items, "num_distinct_items": num_distinct_items,
        "subtotal": subtotal, "min_item_price": min_item_price, "max_item_price": max_item_price,
        "total_onshift_dashers": onshift, "total_busy_dashers": busy, "total_outstanding_orders": outstanding,
        "load_ratio": load_ratio, "busy_ratio": busy_ratio, "demand_pressure": demand_pressure,
        "hour": int(hour), "day_of_week": int(dow), "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday), "peak_hour": int(peak_hour), "is_rush": int(is_rush),
        "estimated_order_place_duration": int(est_place), "estimated_store_to_consumer_driving_duration": int(est_drive),
        "market_id": 1.0, "temperature_c": float(temperature_c), "precip": int(precip),
        # default one-hots to match training (drop='first')
        "weather_Cloudy": 0, "weather_Rain": 0, "weather_Snow": 0,
        "traffic_Light": 0, "traffic_Medium": 1,
        "meal_period_breakfast": 0, "meal_period_dinner": 0, "meal_period_other": 1
    }
    X = pd.DataFrame([row])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    if feature_cols:
        X = X[feature_cols]

    model_choice = st.selectbox(
        "Model",
        [m for m in ["xgb","rf"] if ((m=="xgb" and xgb_model is not None) or (m=="rf" and rf_model is not None))] or ["rf"]
    )

    if st.button("Predict ETA"):
        with st.spinner("Scoring..."):
            mdl = xgb_model if (model_choice=="xgb" and xgb_model is not None) else rf_model
            if mdl is None:
                st.warning("Train a model first (Run prep_and_train_augmented or train_advanced).")
            else:
                eta = float(mdl.predict(X)[0])
                msg = f"ETA: {eta/60:.1f} min ({int(eta)} sec)"
                if q90 is not None:
                    lo, hi = eta - q90, eta + q90
                    msg += f"  •  ± {q90/60:.1f} min  [{lo/60:.1f} – {hi/60:.1f}]"
                st.success(msg)
                try:
                    st.toast("Prediction ready ✅", icon="✅")
                except Exception:
                    pass

# ── Tab: Batch Upload
with tab["Batch Upload"]:
    st.subheader("Batch Upload")
    up = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if up and (xgb_model or rf_model):
        dfu = pd.read_csv(up)
        if feature_cols:
            missing = [c for c in feature_cols if c not in dfu.columns]
            for m in missing:
                dfu[m] = 0
            dfu = dfu[feature_cols]
        mdl = xgb_model or rf_model
        with st.spinner("Scoring batch..."):
            dfu["pred_eta_seconds"] = mdl.predict(dfu)
        st.download_button(
            "Download Predictions",
            dfu.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )
    else:
        st.caption("Train first, then upload a CSV built from data/processed/features_aug.csv columns.")

# ── Tab: Trends
with tab["Trends"]:
    st.subheader("Trends: Error Distribution & Hourly Means")
    try:
        df = pd.read_csv(DATA / "features_aug.csv")
        feature_cols2 = json.loads((ART / "feature_columns_aug.json").read_text())
        model2 = xgb_model or rf_model
        if model2:
            sample = df.sample(n=min(2000, len(df)), random_state=42)
            sample["pred"] = model2.predict(sample[feature_cols2])
            fig = px.histogram(
                (sample["delivery_seconds"] - sample["pred"]) / 60.0,
                nbins=40, title="Error Distribution (minutes)"
            )
            st.plotly_chart(fig, use_container_width=True)
            hourly = sample.copy()
            hourly["hour"] = hourly["hour"].astype(int)
            agg = hourly.groupby("hour")[["delivery_seconds", "pred"]].mean() / 60.0
            st.line_chart(agg.rename(columns={"delivery_seconds": "actual_min", "pred": "pred_min"}))
        else:
            st.info("Train a model to see trends.")
    except Exception:
        st.info("Train first to see trends.")

# ── Tab: Scenario Simulator
with tab["Scenario Simulator"]:
    st.subheader("Scenario Simulator")
    st.write("Adjust inputs in *Single Prediction*, then save snapshots here.")
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []
    can_save = "row" in locals()
    name = st.text_input("Scenario name", "Base Case")
    if st.button("Save current inputs as scenario", disabled=not can_save):
        if can_save:
            st.session_state.scenarios.append((name, row.copy()))
    if st.session_state.scenarios:
        for nm, r in st.session_state.scenarios[-5:]:
            st.json({"name": nm, **{k: r[k] for k in list(r)[:10]}})

# ── Tab: Explainability
with tab["Explainability"]:
    st.subheader("Explainability (SHAP)")
    shap_img = ART / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="SHAP Summary Plot")
    else:
        st.info("Run: `python -m utils.explain_shap` after training to generate SHAP plots.")

# ── Tab: Monitoring (Evidently) — embed + download + sensitivity controls
with tab["Monitoring"]:
    import streamlit.components.v1 as components
    from typing import Optional

    st.subheader("Monitoring & Drift (Evidently)")
    st.caption("Compare a reference window vs a current window in your processed dataset.")
    st.write("Default: first 70% rows = reference; last 30% rows = current. Upload a CSV for a real ‘current’ slice.")

    # Sensitivity knobs (optional)
    c1, c2 = st.columns(2)
    with c1:
        ref_frac = st.slider("Reference fraction (first part of file)", 0.50, 0.95, 0.90, 0.05)
        stattest_threshold = st.slider("Stat test threshold (lower = more sensitive)", 0.01, 0.20, 0.05, 0.01)
    with c2:
        stattest = st.selectbox("Stat test (numeric)", ["wasserstein", "ks", "z"])

    # Optional current-window upload
    recent_upl = st.file_uploader("Optional: upload current-window CSV", type=["csv"], key="upl_cur")
    cur_path: Optional[Path] = None
    if recent_upl:
        tmp = REPORTS / "uploaded_current.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(recent_upl.getbuffer())
        cur_path = tmp

    ref = DATA / "features_aug.csv"
    out = REPORTS / "evidently_drift.html"

    tests_cols = [
        "delivery_seconds","hour","load_ratio",
        "estimated_store_to_consumer_driving_duration",
        "total_outstanding_orders","total_onshift_dashers","total_busy_dashers"
    ]

    if st.button("Generate Drift Report"):
        with st.spinner("Building Evidently report..."):
            try:
                from utils.drift_report import generate_drift_report
                html_path = generate_drift_report(
                    ref_path=ref,
                    cur_path=cur_path,
                    out_path=out,
                    ref_frac=float(ref_frac),
                    tests_cols=tests_cols,
                    stattest=stattest,
                    stattest_threshold=float(stattest_threshold),
                )
                size_kb = html_path.stat().st_size / 1024 if html_path.exists() else 0
                if size_kb < 5:
                    st.warning("Report generated but looks empty; verify your dataset has rows.")
                else:
                    st.success(f"Report generated ({size_kb:.1f} KB).")
            except Exception as e:
                st.error(f"Failed to build report: {e}")

    if out.exists():
        st.markdown("**Preview**")
        try:
            html = out.read_text(encoding="utf-8", errors="ignore")
            components.html(html, height=900, scrolling=True)
        except Exception as e:
            st.info(f"Could not embed report ({e}).")

        st.download_button(
            "Download Evidently report (HTML)",
            data=out.read_bytes(),
            file_name="evidently_drift.html",
            mime="text/html"
        )
        st.caption(f"Saved at: `{out}`")
    else:
        st.info("No report yet. Click **Generate Drift Report** above.")
