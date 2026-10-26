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

def read_user_file(filename: str) -> str:
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise ValueError("Path traversal detected: access denied")
    with open(filepath, "r") as f:
        return f.read()
def read_user_file(filename: str) -> str:
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise ValueError("Path traversal detected: access denied")
    with open(filepath, "r") as f:
        return f.read()
def read_user_file(filename: str) -> str:
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise ValueError("Path traversal detected: access denied")
    with open(filepath, "r") as f:
        return f.read()
def read_user_file(filename: str) -> str:
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise ValueError("Path traversal detected: access denied")
    with open(filepath, "r") as f:
        return f.read()
def read_user_file(filename: str) -> str:
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise ValueError("Path traversal detected: access denied")
    with open(filepath, "r") as f:
        return f.read()
def read_user_file(filename: str) -> str:
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise ValueError("Path traversal detected: access denied")
    with open(filepath, "r") as f:
        return f.read()
def read_user_file(filename: str) -> str:
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise ValueError("Path traversal detected: access denied")
    with open(filepath, "r") as f:
        return f.read()
def read_user_file(filename: str) -> str:
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise ValueError("Path traversal detected: access denied")
    with open(filepath, "r") as f:
        return f.read()
if __name__ == "__main__":
    train_pipeline()
