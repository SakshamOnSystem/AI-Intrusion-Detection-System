import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

DATA_DIR = "CICIDS2017"
COMBINED = os.path.join(DATA_DIR, "combined.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42
PCA_VAR = 0.95
RF_ESTIMATORS = 150
SAMPLE_FRACTION = 0.10   
CHUNK_ROWS = 50000
AE_LATENT = 16

def combine_cic_csvs(folder):
    if os.path.exists(COMBINED):
        print("✅ combined.csv exists")
        return

    print("🔄 Merging CIC-IDS2017 files...")
    dfs = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            fp = os.path.join(folder, f)
            try:
                df = pd.read_csv(fp, low_memory=False, encoding="utf-8")
            except:
                df = pd.read_csv(fp, low_memory=False, encoding="latin1")
            dfs.append(df)

    pd.concat(dfs, ignore_index=True).to_csv(COMBINED, index=False)
    print("✅ combined.csv created")


def load_cic(path, chunksize=CHUNK_ROWS, frac=SAMPLE_FRACTION):
    print("🔁 Loading CIC IDS in memory-safe mode...")

    label_candidates = {"Label", "label", " Label", "Label ", "Attack", " Attack"}
    chunks = []

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding="latin1"):

        lbl = None
        for col in chunk.columns:
            if col in label_candidates or str(col).strip().lower() == "label":
                lbl = col
                break
        
        if lbl is None:
            raise KeyError("❌ Label column not found")

        s = chunk[lbl].astype(str).str.strip()
        s = s.replace({"BENIGN": "0", "Benign": "0"})
        s = s.apply(lambda v: 0 if v == "0" else 1)

        chunk = chunk.drop(columns=[lbl])
        chunk.insert(len(chunk.columns), "Label", s)

        chunk = chunk.sample(frac=frac, random_state=RANDOM_STATE)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded {len(df)} rows (sampled)")
    df = df.dropna(axis=1, how="all")
    return df, "Label"


def preprocess(df, label_col):
    y = df[label_col]
    X = df.drop(columns=[label_col])

    for c in X.columns:
        if X[c].dtype == object:
            try: X[c] = LabelEncoder().fit_transform(X[c].astype(str))
            except: X = X.drop(columns=[c])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.clip(-1e9, 1e9)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    return pd.DataFrame(Xs, columns=X.columns), y.astype(int), scaler

# =============================
# MODELS
# =============================
def pca_reduce(X):
    pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=RANDOM_STATE)
    return pd.DataFrame(pca.fit_transform(X)), pca

def build_autoencoder(dim):
    inp = layers.Input(shape=(dim,))
    x = layers.Dense(64, activation="relu")(inp)
    h = layers.Dense(AE_LATENT, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(h)
    out = layers.Dense(dim)(x)
    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    return ae

def recon_error(ae, X):
    rec = ae.predict(X, verbose=0)
    return np.mean((X - rec) ** 2, axis=1)

def hybrid(clf, ae, X, z=2.5):
    p = clf.predict_proba(X)[:,1]
    r = recon_error(ae, X)
    t = r.mean() + z * r.std()
    y = ((p > 0.5) | (r > t)).astype(int)
    return y, r, t

# =============================
# TRAIN PIPELINE
# =============================
def train_pipeline():
    combine_cic_csvs(DATA_DIR)
    df, label_col = load_cic(COMBINED)

    X, y, scaler = preprocess(df, label_col)

    print("📊 Class distribution:", y.value_counts().to_dict())
    minority_ratio = y.value_counts().min() / y.value_counts().max()

    # ✅ Only apply SMOTE if needed
    if minority_ratio < 0.4:
        print("⚠️ Imbalance detected — applying SMOTE...")
        Xb, yb = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    else:
        print("✅ Balanced — skipping SMOTE")
        Xb, yb = X, y

    print("📉 PCA reducing...")
    Xp, pca = pca_reduce(Xb)

    Xtr, Xte, ytr, yte = train_test_split(Xp, yb, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yb)

    print("🌲 Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(Xtr, ytr)

    print("🤖 Training Autoencoder...")
    ae = build_autoencoder(Xp.shape[1])
    normal = Xtr[ytr == 0]
    ae.fit(normal, normal, epochs=20, batch_size=256, validation_split=0.1,
           callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
           verbose=1)

    print("🧠 Evaluating...")
    yp, r, t = hybrid(clf, ae, Xte)
    print("Threshold:", t)
    print(classification_report(yte, yp))
    print("ROC-AUC:", roc_auc_score(yte, yp))
    print(confusion_matrix(yte, yp))

    print("💾 Saving...")
    joblib.dump(clf, f"{MODEL_DIR}/rf_cic.joblib")
    joblib.dump(pca, f"{MODEL_DIR}/pca_cic.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_cic.joblib")
    ae.save(f"{MODEL_DIR}/autoencoder_cic.keras")
    print("✅ Training complete — models saved!")

if __name__ == "__main__":
    train_pipeline()
