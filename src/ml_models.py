"""
Entrainement et evaluation des modeles Machine Learning :
Random Forest, XGBoost, LightGBM et Ensemble Voting.
Extrait les importances de features normalisees comme poids adaptatifs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


def prepare_ml_data(df: pd.DataFrame,
                    feature_cols: list[str],
                    test_size: float = 0.3,
                    random_state: int = 42):
    """
    Prepare et divise le dataset pour l'entrainement ML.

    Retourne (X_train, X_test, y_train, y_test, scaler).
    """
    X = df[feature_cols].values
    y = df['inonde'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, scaler, X_scaled


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, name: str) -> dict:
    """Calcule AUC, F1, precision, recall et matrice de confusion."""
    y_pred = (y_proba > 0.5).astype(int)
    metrics = {
        'modele':    name,
        'auc':       roc_auc_score(y_true, y_proba),
        'f1':        f1_score(y_true, y_pred),
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'cm':        confusion_matrix(y_true, y_pred),
    }
    return metrics


def train_random_forest(X_train, y_train, random_state: int = 42):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train, X_test, y_test, random_state: int = 42):
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, random_state: int = 42):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=8,
        is_unbalance=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.log_evaluation(period=0)])
    return model


def train_ensemble(rf, xgb_model, lgb_model, X_train, y_train):
    """Ensemble soft voting des 3 modeles deja entraines."""
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model), ('lgb', lgb_model)],
        voting='soft',
        weights=[1, 1, 1],
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def extract_ml_weights(rf, xgb_model, lgb_model,
                       feature_cols: list[str]) -> dict[str, float]:
    """
    Moyenne des importances de features des 3 modeles, normalisee a 1.
    Permet de deriver des poids 'data-driven' equivalents aux poids AHP.
    """
    imp_rf  = rf.feature_importances_  / rf.feature_importances_.sum()
    imp_xgb = xgb_model.feature_importances_ / xgb_model.feature_importances_.sum()
    imp_lgb = lgb_model.feature_importances_ / lgb_model.feature_importances_.sum()
    avg = (imp_rf + imp_xgb + imp_lgb) / 3
    return dict(zip(feature_cols, avg))


def train_all_models(df: pd.DataFrame,
                     feature_cols: list[str],
                     models_dir: Path,
                     test_size: float = 0.3,
                     random_state: int = 42) -> dict:
    """
    Pipeline complet : preparation, entrainement des 4 modeles, evaluation,
    sauvegarde dans models_dir.

    Retourne un dict avec : modeles, metriques, poids ML, scaler, X_scaled.
    """
    print("\n=== ENTRAINEMENT MODELES ML ===")
    X_train, X_test, y_train, y_test, scaler, X_scaled = prepare_ml_data(
        df, feature_cols, test_size, random_state
    )
    print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

    print("\n[1/4] Random Forest...")
    rf = train_random_forest(X_train, y_train, random_state)
    metrics_rf = evaluate_model(y_test, rf.predict_proba(X_test)[:, 1], "Random Forest")
    print(f"  AUC={metrics_rf['auc']:.3f}  F1={metrics_rf['f1']:.3f}")

    print("[2/4] XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test, random_state)
    metrics_xgb = evaluate_model(y_test, xgb_model.predict_proba(X_test)[:, 1], "XGBoost")
    print(f"  AUC={metrics_xgb['auc']:.3f}  F1={metrics_xgb['f1']:.3f}")

    print("[3/4] LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test, random_state)
    metrics_lgb = evaluate_model(y_test, lgb_model.predict_proba(X_test)[:, 1], "LightGBM")
    print(f"  AUC={metrics_lgb['auc']:.3f}  F1={metrics_lgb['f1']:.3f}")

    print("[4/4] Ensemble Voting...")
    ensemble = train_ensemble(rf, xgb_model, lgb_model, X_train, y_train)
    metrics_ens = evaluate_model(y_test, ensemble.predict_proba(X_test)[:, 1], "Ensemble")
    print(f"  AUC={metrics_ens['auc']:.3f}  F1={metrics_ens['f1']:.3f}")

    # Poids ML (feature importances moyennees)
    poids_ml_cols = extract_ml_weights(rf, xgb_model, lgb_model, feature_cols)

    # Sauvegarde
    models_dir.mkdir(exist_ok=True)
    pickle.dump(rf,        open(models_dir / "rf_model.pkl",       "wb"))
    pickle.dump(xgb_model, open(models_dir / "xgb_model.pkl",      "wb"))
    pickle.dump(lgb_model, open(models_dir / "lgb_model.pkl",      "wb"))
    pickle.dump(ensemble,  open(models_dir / "ensemble_model.pkl", "wb"))
    pickle.dump(scaler,    open(models_dir / "scaler.pkl",         "wb"))
    pickle.dump(feature_cols, open(models_dir / "feature_cols.pkl", "wb"))
    print(f"\nModeles sauvegardes dans {models_dir}/")

    all_metrics = [metrics_rf, metrics_xgb, metrics_lgb, metrics_ens]
    summary_df = pd.DataFrame([
        {'Modele': m['modele'], 'AUC-ROC': m['auc'], 'F1-Score': m['f1'],
         'Precision': m['precision'], 'Recall': m['recall']}
        for m in all_metrics
    ])
    print("\n" + summary_df.to_string(index=False))

    return {
        'rf': rf, 'xgb': xgb_model, 'lgb': lgb_model, 'ensemble': ensemble,
        'scaler': scaler, 'X_scaled': X_scaled,
        'metrics': all_metrics, 'summary': summary_df,
        'poids_ml_cols': poids_ml_cols,
        'X_test': X_test, 'y_test': y_test,
    }
