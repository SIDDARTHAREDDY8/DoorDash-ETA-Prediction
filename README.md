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

---

## ⚡ Getting Started

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DoorDash-ETA-Prediction.git
cd DoorDash-ETA-Prediction


python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

streamlit run app/app.py

python -m utils.explain_shap

python -m utils.drift_report

[theme]
base = "light"

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


---



