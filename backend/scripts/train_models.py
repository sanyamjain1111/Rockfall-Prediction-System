# backend/scripts/train_models.py
# ------------------------------------------------------------
# Full end-to-end trainer for RF, XGBoost, CNN, LSTM + CV-stacked ensemble.
# - Uses tabular CSV: data/processed/training_data.csv
# - Optional geospatial stack: data/processed/geospatial_stack.npy (CNN)
# - Optional time-series: data/processed/timeseries.csv (LSTM)
# - Saves artifacts into models/saved_models/
# - Writes TRAINING_REPORT.json with AUCs/components
#
# Usage examples:
#   python scripts/train_models.py
#   python scripts/train_models.py --allow-synthetic-lstm
#   python scripts/train_models.py --no-cnn --no-lstm
# ------------------------------------------------------------

import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Paths & Globals ----------
ROOT = Path(__file__).resolve().parents[1]  # backend/
DATA_DIR = ROOT / "data"
SAVE_DIR = ROOT / "models" / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TABULAR_CSV = DATA_DIR / "processed" / "training_data.csv"
GEO_STACK = DATA_DIR / "processed" / "geospatial_stack.npy"  # (H,W,B)
TS_CSV = DATA_DIR / "processed" / "timeseries.csv"           # zone_id,date/rainfall/temperature/insar_velocity,label

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------- Utilities ----------
def log(msg: str):
    print(msg, flush=True)


def safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


# ---------- Tabular Data ----------
def load_tabular():
    assert TABULAR_CSV.exists(), f"Missing {TABULAR_CSV}"
    df = pd.read_csv(TABULAR_CSV)
    df.columns = [c.strip().lower() for c in df.columns]

    features = [
        "slope_angle", "aspect", "elevation", "plan_curvature", "profile_curvature",
        "twi", "spi", "lithology", "distance_to_fault", "fracture_density", "cohesion",
        "friction_angle", "unit_weight", "pore_pressure_ratio",
        "cumulative_rainfall_24h", "cumulative_rainfall_72h", "temperature_range",
        "days_since_blast", "insar_velocity"
    ]
    label_col = "label"

    missing = [c for c in features + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.dropna(subset=features + [label_col])

    # Encode lithology if needed
    enc = None
    if df["lithology"].dtype == object:
        enc = LabelEncoder()
        df["lithology"] = enc.fit_transform(df["lithology"])
        joblib.dump(enc, SAVE_DIR / "lithology_encoder.pkl")
        log(f"[Tabular] Saved lithology encoder: {SAVE_DIR / 'lithology_encoder.pkl'}")

    X = df[features].to_numpy(dtype=float)
    y = df[label_col].astype(int).to_numpy()
    return X, y


# ---------- Models: RF / XGB ----------
def train_rf(X, y):
    log("[RF] Training RandomForest...")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced', random_state=RANDOM_STATE
    )
    rf.fit(Xtr, ytr)
    auc = safe_auc(yte, rf.predict_proba(Xte)[:, 1])
    joblib.dump(rf, SAVE_DIR / "random_forest.pkl")
    log(f"[RF] Saved: {SAVE_DIR / 'random_forest.pkl'} | AUC={auc}")
    return rf, auc


def train_xgb(X, y):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        log(f"[XGB] xgboost not available: {e} — skipping.")
        return None, None

    log("[XGB] Training XGBoost (tuned)...")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # class imbalance handling
    pos = (ytr == 1).sum()
    neg = (ytr == 0).sum()
    spw = float(neg / max(pos, 1))

    xgb = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        gamma=0.0,
        scale_pos_weight=spw,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE
    )

    xgb.fit(
        Xtr, ytr,
        eval_set=[(Xte, yte)],
        verbose=False,
        early_stopping_rounds=50
    )

    # Use best iteration if available
    best_ntree = getattr(xgb, "best_ntree_limit", None)
    if best_ntree:
        proba = xgb.predict_proba(Xte, ntree_limit=best_ntree)[:, 1]
    else:
        proba = xgb.predict_proba(Xte)[:, 1]

    auc = safe_auc(yte, proba)
    joblib.dump(xgb, SAVE_DIR / "xgboost.pkl")
    log(f"[XGB] Saved: {SAVE_DIR / 'xgboost.pkl'} | AUC={auc}")
    return xgb, auc


# ---------- CNN from Geo Stack ----------
def make_geo_labels_from_stack(stack: np.ndarray) -> np.ndarray:
    """
    Heuristic labels if no ground-truth raster:
    HIGH risk if (slope>35 & twi>12) OR spi above 85th percentile.
    Bands assumed: [elev, slope, aspect, plan, profile, twi, spi, flowacc]
    """
    slope = stack[..., 1]
    twi = stack[..., 5]
    spi = stack[..., 6]
    risk = ((slope > 35) & (twi > 12)) | (spi > np.percentile(spi, 85))
    return risk.astype(np.uint8)


def sample_patches(stack: np.ndarray, y_map: np.ndarray, patch: int, n: int):
    H, W, B = stack.shape
    r = patch // 2
    xs, ys = [], []
    for _ in range(n):
        i = np.random.randint(r, H - r)
        j = np.random.randint(r, W - r)
        xs.append(stack[i - r:i + r + 1, j - r:j + r + 1, :])
        ys.append(y_map[i, j])
    Xp = np.stack(xs).astype(np.float32)
    yp = np.array(ys).astype(np.uint8)

    # Per-band min-max normalization
    flat = Xp.reshape(-1, Xp.shape[-1])
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    rng = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    Xp = (Xp - mins) / rng
    return Xp, yp


def train_cnn_from_stack(patch=5, n_patches=12000, epochs=8, batch_size=256):
    if not GEO_STACK.exists():
        log("[CNN] geospatial_stack.npy not found — skipping.")
        return None, None
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as e:
        log(f"[CNN] TensorFlow not available: {e} — skipping.")
        return None, None

    log("[CNN] Loading geospatial stack...")
    stack = np.load(GEO_STACK)  # (H, W, B)
    y_map = make_geo_labels_from_stack(stack)
    Xp, yp = sample_patches(stack, y_map, patch=patch, n=n_patches)

    # Split
    idx = np.arange(len(yp))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, va = idx[:split], idx[split:]
    Xtr, ytr, Xva, yva = Xp[tr], yp[tr], Xp[va], yp[va]

    log("[CNN] Building model...")
    model = models.Sequential([
        layers.Input(shape=(patch, patch, Xp.shape[-1])),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    log("[CNN] Training...")
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size, verbose=2)

    # Quick AUC on val
    p = model.predict(Xva, verbose=0).ravel()
    auc = safe_auc(yva, p)
    out_path = SAVE_DIR / "cnn_model.h5"
    model.save(out_path)
    log(f"[CNN] Saved: {out_path} | AUC={auc}")
    return str(out_path.name), auc


# ---------- LSTM from Time Series ----------
def train_lstm_from_timeseries(allow_synth=False, epochs=5, batch_size=128):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as e:
        log(f"[LSTM] TensorFlow not available: {e} — skipping.")
        return None, None

    if TS_CSV.exists():
        log("[LSTM] Loading timeseries.csv...")
        ts = pd.read_csv(TS_CSV)
        lc = {c.lower(): c for c in ts.columns}
        needed = {"zone_id", "rainfall", "temperature", "insar_velocity", "label"}
        if not needed.issubset(set(lc.keys())):
            log("[LSTM] timeseries.csv present but columns mismatch — skipping.")
            return None, None

        # Sort by zone and time column if present
        time_key = lc.get("date", lc.get("day"))
        if time_key:
            ts = ts.sort_values([lc["zone_id"], time_key])

        Xs, ys = [], []
        for _, g in ts.groupby(lc["zone_id"]):
            arr = g[[lc["rainfall"], lc["temperature"], lc["insar_velocity"]]].to_numpy(float)
            lab = g[lc["label"]].astype(int).to_numpy()
            if arr.shape[0] >= 30:
                for start in range(0, arr.shape[0] - 30 + 1):
                    Xs.append(arr[start:start + 30])
                    ys.append(int(np.round(lab[start:start + 30].mean())))
        if not Xs:
            log("[LSTM] Not enough steps to build sequences — skipping.")
            return None, None
        X = np.stack(Xs).astype(np.float32)
        y = np.array(ys).astype(np.uint8)
    elif allow_synth:
        log("[LSTM] Synthesizing demo sequences...")
        X = np.random.rand(6000, 30, 3).astype(np.float32)
        score = (0.4 * X[:, :, 0].mean(axis=1)
                 + 0.3 * X[:, :, 2].mean(axis=1)
                 + 0.2 * (1.0 - np.abs(X[:, :, 1] - 0.5).mean(axis=1)))
        y = (score > np.quantile(score, 0.6)).astype(np.uint8)
    else:
        log("[LSTM] No timeseries.csv and no --allow-synthetic-lstm — skipping.")
        return None, None

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, va = idx[:split], idx[split:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    log("[LSTM] Building model...")
    model = models.Sequential([
        layers.Input(shape=(30, 3)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    log("[LSTM] Training...")
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size, verbose=2)

    p = model.predict(Xva, verbose=0).ravel()
    auc = safe_auc(yva, p)
    out_path = SAVE_DIR / "lstm_model.h5"
    model.save(out_path)
    log(f"[LSTM] Saved: {out_path} | AUC={auc}")
    return str(out_path.name), auc


# ---------- Proper CV Stacking (no leakage) ----------
def train_ensemble_cv(rf, xgb, X, y, n_splits=5, random_state=RANDOM_STATE):
    """
    Proper stacking:
      1) Split X/y into train/test.
      2) On train: build OOF predictions for each base model via CV.
      3) Fit meta-learner on OOF.
      4) Retrain base models on all train.
      5) Evaluate on test using stacked base predictions.
    Saves 'ensemble_stacker.pkl' with {'type','models','meta','components'}.
    """
    base_models = {}
    components = []
    if rf is None and xgb is None:
        log("[Ensemble] No base models — skipping.")
        return None, None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    def oof_preds(estimator, name):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof = np.zeros(len(y_tr), dtype=float)
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            est = clone(estimator)
            est.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx] = est.predict_proba(X_tr[va_idx])[:, 1]
        # retrain on full train
        est_full = clone(estimator)
        est_full.fit(X_tr, y_tr)
        base_models[name] = est_full
        components.append(name)
        return oof[:, None]

    Z_parts = []
    if rf is not None:
        Z_parts.append(oof_preds(rf, "rf"))
    if xgb is not None:
        Z_parts.append(oof_preds(xgb, "xgb"))

    if not Z_parts:
        log("[Ensemble] No base models after OOF — skipping.")
        return None, None

    Z_tr = np.hstack(Z_parts)
    meta = LogisticRegression(max_iter=200)
    meta.fit(Z_tr, y_tr)

    # Evaluate on test
    Z_te_parts = []
    for name in components:
        Z_te_parts.append(base_models[name].predict_proba(X_te)[:, 1][:, None])
    Z_te = np.hstack(Z_te_parts)
    ens_auc = safe_auc(y_te, meta.predict_proba(Z_te)[:, 1])

    # Save
    obj = {"type": "stacked", "models": base_models, "meta": meta, "components": components}
    joblib.dump(obj, SAVE_DIR / "ensemble_stacker.pkl")
    log(f"[Ensemble] Saved: {SAVE_DIR / 'ensemble_stacker.pkl'} | components={components} | AUC={ens_auc}")
    return "stacked", ens_auc


def train_calibrated_single(estimator, X, y):
    """
    If only one base model available, fit a calibrated version as 'ensemble' for uniform API.
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    calib = CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=3)
    calib.fit(Xtr, ytr)
    auc = safe_auc(yte, calib.predict_proba(Xte)[:, 1])
    obj = {"type": "calibrated", "model": calib, "components": ["single"]}
    joblib.dump(obj, SAVE_DIR / "ensemble_stacker.pkl")
    log(f"[Ensemble] Saved calibrated single as ensemble | AUC={auc}")
    return "calibrated", auc


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-xgb", action="store_true", help="Skip XGBoost")
    ap.add_argument("--no-cnn", action="store_true", help="Skip CNN")
    ap.add_argument("--no-lstm", action="store_true", help="Skip LSTM")
    ap.add_argument("--allow-synthetic-lstm", action="store_true",
                    help="If no timeseries.csv, synthesize demo sequences")
    ap.add_argument("--cnn-patches", type=int, default=12000, help="Number of CNN patches to sample")
    ap.add_argument("--cnn-epochs", type=int, default=8, help="CNN training epochs")
    ap.add_argument("--cnn-batch", type=int, default=256, help="CNN batch size")
    ap.add_argument("--lstm-epochs", type=int, default=5, help="LSTM training epochs")
    ap.add_argument("--lstm-batch", type=int, default=128, help="LSTM batch size")
    args = ap.parse_args()

    summary = {}

    # Tabular load
    X, y = load_tabular()
    log(f"[Tabular] Loaded: X={X.shape}, positives={int(y.sum())}/{len(y)} ({y.mean():.2%})")

    # RF
    rf, rf_auc = train_rf(X, y)
    summary["random_forest_auc"] = rf_auc

    # XGB (optional)
    xgb, xgb_auc = (None, None)
    if not args.no_xgb:
        xgb, xgb_auc = train_xgb(X, y)
        if xgb is not None:
            summary["xgboost_auc"] = xgb_auc

    # CNN (optional)
    cnn_file, cnn_auc = (None, None)
    if not args.no_cnn:
        cnn_file, cnn_auc = train_cnn_from_stack(
            patch=5, n_patches=args.cnn_patches,
            epochs=args.cnn_epochs, batch_size=args.cnn_batch
        )
        if cnn_file:
            summary["cnn_auc"] = cnn_auc

    # LSTM (optional)
    lstm_file, lstm_auc = (None, None)
    if not args.no_lstm:
        lstm_file, lstm_auc = train_lstm_from_timeseries(
            allow_synth=args.allow_synthetic_lstm,
            epochs=args.lstm_epochs, batch_size=args.lstm_batch
        )
        if lstm_file:
            summary["lstm_auc"] = lstm_auc

    # Ensemble
    if xgb is not None:
        ens_type, ens_auc = train_ensemble_cv(rf, xgb, X, y)
    else:
        ens_type, ens_auc = train_calibrated_single(rf, X, y)
    summary["ensemble_type"] = ens_type
    if ens_auc is not None:
        summary["ensemble_auc"] = ens_auc

    # Report
    report_path = SAVE_DIR / "TRAINING_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[Report] Wrote: {report_path}")
    log("== Training summary ==")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
