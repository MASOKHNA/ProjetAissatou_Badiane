"""
Modele LSTM pour la prediction temporelle des inondations.
Prend en entree une fenetre de 6 mois et predit le mois suivant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Reproductibilite
tf.random.set_seed(42)
np.random.seed(42)

FEATURES_TS = ['pluie', 'altitude', 'pente', 'drainage', 'sol']


def normalize_timeseries(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Normalise les features TS dans [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df = df.copy()
    df[FEATURES_TS] = scaler.fit_transform(df[FEATURES_TS])
    return df, scaler


def make_sequences(data: np.ndarray, lookback: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforme un tableau (timesteps, features+label) en sequences LSTM.
    La colonne 0 du tableau est le label (inonde).

    Retourne X (samples, lookback, n_features) et y (samples,).
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)


def prepare_lstm_data(df_ts: pd.DataFrame,
                      lookback: int = 6,
                      train_ratio: float = 0.8) -> dict:
    """
    Normalise le dataset temporel, construit les sequences et divise en
    train / test en respectant l'ordre chronologique.
    """
    df_ts, scaler = normalize_timeseries(df_ts)

    X_list, y_list = [], []
    for zone_id in df_ts['zone_id'].unique():
        zone_df = df_ts[df_ts['zone_id'] == zone_id].sort_values('date')
        label_col = zone_df['inonde'].values.reshape(-1, 1).astype(float)
        feat_cols = zone_df[FEATURES_TS].values
        combined  = np.hstack([label_col, feat_cols])
        X_seq, y_seq = make_sequences(combined, lookback)
        X_list.append(X_seq)
        y_list.append(y_seq)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    split = int(train_ratio * len(X))
    return {
        'X_train': X[:split], 'y_train': y[:split],
        'X_test':  X[split:], 'y_test':  y[split:],
        'n_features': X.shape[2],
        'scaler': scaler,
    }


def build_lstm_model(lookback: int, n_features: int) -> Sequential:
    model = Sequential([
        LSTM(64, activation='relu',
             input_shape=(lookback, n_features),
             return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC'],
    )
    return model


def train_lstm(df_ts: pd.DataFrame,
               lookback: int = 6,
               epochs: int = 50,
               batch_size: int = 32,
               models_dir: Path | None = None) -> dict:
    """
    Pipeline complet LSTM : preparation -> construction -> entrainement -> evaluation.
    Sauvegarde le modele dans models_dir si fourni.
    """
    print("\n=== ENTRAINEMENT LSTM ===")
    data = prepare_lstm_data(df_ts, lookback=lookback)

    X_train, y_train = data['X_train'], data['y_train']
    X_test,  y_test  = data['X_test'],  data['y_test']
    n_features = data['n_features']

    print(f"X_train : {X_train.shape}  |  X_test : {X_test.shape}")

    model = build_lstm_model(lookback, n_features)
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = model.predict(X_test, verbose=0).ravel()
    auc  = roc_auc_score(y_test, y_pred)
    f1   = f1_score(y_test, (y_pred > 0.5).astype(int), zero_division=0)

    print(f"\nResultats LSTM -> AUC={auc:.3f}  F1={f1:.3f}")

    if models_dir is not None:
        models_dir.mkdir(exist_ok=True)
        model.save(models_dir / "lstm_model.keras")
        print(f"Modele LSTM sauvegarde : {models_dir / 'lstm_model.keras'}")

    return {
        'model': model,
        'history': history,
        'auc': auc,
        'f1': f1,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
    }
