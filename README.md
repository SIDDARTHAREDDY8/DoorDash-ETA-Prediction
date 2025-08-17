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

# 🚀 DoorDash ETA Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://doordash-eta-prediction-hox7vaedsb6tumugrgx38e.streamlit.app)

An end-to-end **machine learning project** that predicts **Estimated Time of Arrival (ETA)** for DoorDash-style deliveries.  
The project integrates **prediction, explainability (SHAP), and monitoring (Evidently)** in an interactive **Streamlit dashboard**.

---

## ✨ Features

✅ **Single Prediction** – enter order details and get instant ETA predictions  
✅ **Batch Prediction** – upload CSVs for bulk predictions with downloadable results  
✅ **Explainability (SHAP)** – visualize feature importance and model decisions  
✅ **Monitoring (Evidently)** – detect and analyze data drift using live or uploaded datasets  
✅ **Auto-training** – if no trained model exists, a compact Random Forest is trained automatically  
✅ **Streamlit UI** – user-friendly interface with plots, downloads, and interactive insights  

---

## 🌐 Live Demo

👉 [**Click here to try the app**](https://doordash-eta-prediction-hox7vaedsb6tumugrgx38e.streamlit.app)

---

## 🧱 Tech Stack

- **Backend/ML**: Python, scikit-learn, joblib  
- **Visualization**: Plotly, Matplotlib  
- **Explainability**: SHAP  
- **Monitoring**: Evidently  
- **Frontend/UI**: Streamlit  
- **Hosting**: Streamlit Community Cloud  

---

## 📂 Project Structure

```bash
.
├── app/
│   └── app.py                     # Main Streamlit app (light theme)
├── utils/
│   ├── bootstrap.py               # Auto-trains a compact RF if artifacts are missing
│   ├── explain_shap.py            # Generates SHAP plots
│   └── drift_report.py            # Builds Evidently drift report
├── data/
│   ├── raw/
│   │   └── historical_data.csv    # (Optional) raw data for first-run training
│   └── processed/
│       └── features_aug.csv       # Processed features + target (optional but recommended)
├── artifacts_aug/                 # Saved model, features list, SHAP images, metadata
├── reports/                       # Drift report HTML and uploads
├── requirements.txt
└── .streamlit/
    └── config.toml                # Light theme config
```
---

# 🧪 Run Locally
- Requires **Python 3.10–3.12** recommended (3.9 also works).

```bash
# 1) Clone
git clone <your-repo-url>
cd <your-repo-folder>

# 2) Create & activate virtual env
python3 -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Put data for first run
#    - data/processed/features_aug.csv  (preferred)
#    - OR data/raw/historical_data.csv

# 5) Launch the app
streamlit run app/app.py
```
On first run (if artifacts are missing), the app will train a compact model and cache artifacts under artifacts_aug/.

---
# 🧠 Training / Artifacts
- **Auto-train (preferred on cloud)**: Happens automatically when the app starts and no artifacts exist.
- **Manual train (optional)**: If you have a separate training script, save:
  - **Model** → artifacts_aug/rf_model_aug.joblib
  - **Features list** → artifacts_aug/feature_columns_aug.json
  - **Metadata** → artifacts_aug/model_metadata.json
  - **(Optional) Conformal file** → artifacts_aug/conformal_q90.npy
- Keep models small for free tiers **(fewer trees, joblib.dump(..., compress=3))**.

---

# 🔍 Explainability (SHAP)
- **The app** has a Generate / Refresh SHAP plots button (Explainability tab).
- **You can also run via CLI**:
```bash
python -m utils.explain_shap
```
**Outputs:**
- **artifacts_aug/shap_summary.png** (beeswarm)
- **artifacts_aug/shap_bar.png** (top features by mean |SHAP|)
- **artifacts_aug/shap_top_features.csv**

---

# 📉 Monitoring (Evidently)
In the Monitoring tab:
- **Default:** reference = first 70% of data/processed/features_aug.csv, current = last 30%
-Or **upload a “current” CSV** (same schema) to compare
You can also generate from CLI if you prefer:
```bash
python -m utils.drift_report
```
**Output**: reports/evidently_drift.html (embedded & downloadable in the app)

---

# ⚙️ Configuration

**Theme**: Light only (via .streamlit/config.toml)

```bash
[theme]
base = "light"
```

---

# 🧯 Troubleshooting

- **App says “No model loaded” or “Train a model first”**
  - **Ensure at least one of**:
	  - **data/processed/features_aug.csv** exists in the repo, or
    - **data/raw/historical_data.csv** exists in the repo
  - The **app will auto-train** a compact model on first run if data is present.

- **SHAP error or blank plots**
  - Check that a model exists under **artifacts_aug/ and that requirements.txt** includes shap and matplotlib.
  - Use the button **“Generate / Refresh SHAP plots”** in the app.

- **Evidently report empty**
  -Make sure **data/processed/features_aug.csv** has rows and includes delivery_seconds + feature columns.

- **Large files warning on GitHub**
  - Don’t commit **huge datasets or models**. Keep only a small demo CSV; download large files at runtime if needed.

⸻

# 📦 Requirements (minimum)
```bash
streamlit>=1.28
pandas
numpy
scikit-learn
plotly>=5,<6
jinja2>=3.1,<4
evidently>=0.6.0
joblib
shap
matplotlib
```
---

# 🙌 Acknowledgements
  - **DoorDash-style** open datasets for research/education
  - **Streamlit** for the app framework
  - **SHAP** for explainability
  - **Evidently88 for monitoring
---
**Run the app:**

```bash
cd app
streamlit run app.py
```
---

## 👤 Author

**[Siddartha Reddy Chinthala](https://www.linkedin.com/in/siddarthareddy9)**  
🎓 Master’s in CS | Aspiring Data Scientist  
🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/siddarthareddy9)

⭐️ Show Some Love
If you like this project, don’t forget to ⭐️ the repo and share it!

