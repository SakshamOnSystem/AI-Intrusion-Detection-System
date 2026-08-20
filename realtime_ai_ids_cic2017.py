import os
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model


DATA_DIR = "CICIDS2017"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "realtime_alerts.log")


STREAM_FILE = "CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"

MAX_ROWS = 2000

print("🔹 Loading models...")
rf = joblib.load("models/rf_cic.joblib")
pca = joblib.load("models/pca_cic.joblib")
scaler = joblib.load("models/scaler_cic.joblib")
ae = load_model("models/autoencoder_cic.keras")
print("✅ Models loaded.\n")

print(f"📡 Streaming from: {STREAM_FILE}")
df_stream = pd.read_csv(STREAM_FILE, low_memory=False)

expected_cols = scaler.feature_names_in_

def log_alert(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

print("🚀 Starting live IDS...\n")


for i, row in df_stream.iterrows():


    X = row.to_frame().T

    X = X.drop(columns=[c for c in X.columns if c.lower() == "label"], errors="ignore")

    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0


    X = X[expected_cols]


    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())   
    X = X.fillna(0)            

    
    X = X.clip(-1e9, 1e9)


    Xs = scaler.transform(X)
    Xp = pca.transform(Xs)

    
    prob = rf.predict_proba(Xp)[0][1]
    recon = ae.predict(Xp, verbose=0)
    err = np.mean((Xp - recon) ** 2)

    threshold = err.mean() + 2.5 * err.std()
    is_attack = (prob > 0.50) or (err > threshold)

    if is_attack:
        msg = f"🚨 ALERT | Row {i} | RF={prob:.3f} | AE_ERR={err:.6f}"
        print(msg)
        log_alert(msg)
    else:
        print(f"✅ Normal | Row {i}")

    if i >= MAX_ROWS:
        print("\n✅ Demo finished — Stopped at 2000 rows.")
        break

    time.sleep(0.02) 
