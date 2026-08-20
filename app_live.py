# app_live.py
# Streamlit dashboard for AI-IDS (demo-attack-rich mode)
# - Uses PortScan afternoon file (attack-rich)
# - Loads a direct attack slice (20000:40000) to guarantee alerts
# - Lower demo threshold (0.4) for visible alerts

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

MODEL_DIR = "models"

DATA_FILE = "CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
BATCH_SIZE = 300
REFRESH_INTERVAL = 5000  
ALERT_THRESHOLD = 0.4    
SLICE_START = 20000      
SLICE_END = 40000       

@st.cache_resource
def load_models():
    rf = joblib.load(f"{MODEL_DIR}/rf_cic.joblib")
    pca = joblib.load(f"{MODEL_DIR}/pca_cic.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/scaler_cic.joblib")
    ae = tf.keras.models.load_model(f"{MODEL_DIR}/autoencoder_cic.keras")
    expected_cols = scaler.feature_names_in_
    return rf, pca, scaler, ae, expected_cols

rf, pca, scaler, ae, expected_cols = load_models()

def preprocess_batch(df):
    df = df.copy()

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])


    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    df = df.fillna(0)

    X_scaled = scaler.transform(df)
    X_pca = pca.transform(X_scaled)
    return pd.DataFrame(X_pca)

def compute_reconstruction_error(ae, X):
    recon = ae.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=1)

def hybrid_score(clf, ae, X, z=2.5):
    probs = clf.predict_proba(X)[:, 1]
    rec_err = compute_reconstruction_error(ae, X)
    thr = rec_err.mean() + z * rec_err.std()
    anomaly = np.clip(rec_err / thr, 0, 1)
    hybrid = 0.7*probs + 0.3*anomaly
    return hybrid, probs, anomaly


st.set_page_config(page_title="AI-IDS Live Dashboard (Demo: PortScan)", layout="wide")
st.title("🛡 AI-IDS Live Intrusion Detection")
st.markdown(
    f"Dataset: **PortScan (Afternoon)** — displaying rows **{SLICE_START}:{SLICE_END}**. "
    f"Alert threshold = **{ALERT_THRESHOLD}** (demo sensitivity)."
)


st_autorefresh(interval=REFRESH_INTERVAL, key="refresh_key")

@st.cache_data
def load_attack_slice(path, start, end):
    df = pd.read_csv(path, low_memory=False)

    start = max(0, min(start, len(df)-1))
    end = max(start+1, min(end, len(df)))
    return df.iloc[start:end].reset_index(drop=True)


df_full = load_attack_slice(DATA_FILE, SLICE_START, SLICE_END)

if "cursor" not in st.session_state:
    st.session_state.cursor = 0
if "alert_log" not in st.session_state:
    st.session_state.alert_log = pd.DataFrame(columns=["Index","Hybrid","RF","Anomaly"])

start = st.session_state.cursor
end = min(start + BATCH_SIZE, len(df_full))
batch = df_full.iloc[start:end]
st.session_state.cursor = 0 if end == len(df_full) else end


X = preprocess_batch(batch)
hybrid, probs, anomaly = hybrid_score(rf, ae, X)
alerts = hybrid > ALERT_THRESHOLD

new_alerts = pd.DataFrame({
    "Index": batch.index + SLICE_START,  
    "Hybrid": hybrid,
    "RF": probs,
    "Anomaly": anomaly
})[alerts]

st.session_state.alert_log = pd.concat([st.session_state.alert_log, new_alerts]).tail(1000)


col1, col2, col3 = st.columns(3)
col1.metric("Processed Samples (this session)", end)
col2.metric("Detected Alerts (stored)", len(st.session_state.alert_log))
col3.metric("Refresh (s)", REFRESH_INTERVAL/1000)


fig = px.line(x=list(range(len(hybrid))), y=hybrid, title="Hybrid Threat Score (Current Batch)")
fig.update_xaxes(title="Sample Index (batch)")
fig.update_yaxes(title="Hybrid Score")
st.plotly_chart(fig, use_container_width=True)


st.subheader("⚠️ Live Attack Alerts (Top)")
if len(st.session_state.alert_log) > 0:
    st.dataframe(st.session_state.alert_log.sort_values("Hybrid", ascending=False).head(20))
else:
    st.info("✅ No alerts detected yet in displayed window — try waiting a refresh.")

csv = st.session_state.alert_log.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Alert Log", csv, "ai_ids_live_log.csv", "text/csv")
