"""
Fusion hybride AHP + ML :
- Compare les poids experts (AHP) et les poids data-driven (ML)
- Produit des poids finaux combines (alpha * AHP + (1-alpha) * ML)
- Valide les trois approches vs zones inondees observees
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix)

import config
from src.ahp import apply_ahp_weights, classify_susceptibility


# Mapping colonnes normalisees -> nom de critere AHP
_COL_TO_CRIT = {
    'altitude_norm': 'altitude',
    'pente_norm':    'pente',
    'pluie_norm':    'pluie',
    'drainage_norm': 'drainage',
    'sol_norm':      'sol',
    'usage_norm':    'usage',
}


def ml_weights_to_ahp_names(poids_ml_cols: dict[str, float]) -> dict[str, float]:
    """Traduit les cles feature_col -> nom critere AHP."""
    return {_COL_TO_CRIT[col]: w for col, w in poids_ml_cols.items()}


def compute_hybrid_weights(poids_ahp: dict[str, float],
                            poids_ml: dict[str, float],
                            alpha_expert: float = 0.70) -> dict[str, float]:
    """
    Fusionne poids AHP et poids ML via une combinaison lineaire.
    alpha_expert * AHP + (1 - alpha_expert) * ML, normalise a 1.
    """
    alpha_data = 1.0 - alpha_expert
    combined = {
        crit: alpha_expert * poids_ahp[crit] + alpha_data * poids_ml[crit]
        for crit in poids_ahp
    }
    total = sum(combined.values())
    return {k: v / total for k, v in combined.items()}


def compare_weights(poids_ahp: dict[str, float],
                    poids_ml: dict[str, float]) -> pd.DataFrame:
    """
    Retourne un DataFrame comparant AHP, ML et leur ecart,
    trie par valeur absolue d'ecart decroissante.
    """
    rows = []
    for crit in poids_ahp:
        ahp_pct = poids_ahp[crit] * 100
        ml_pct  = poids_ml[crit] * 100
        rows.append({
            'Critere':         crit,
            'AHP Expert (%)':  round(ahp_pct, 2),
            'ML Data (%)':     round(ml_pct, 2),
            'Ecart (%)':       round(ml_pct - ahp_pct, 2),
        })
    df = pd.DataFrame(rows)
    df = df.reindex(df['Ecart (%)'].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True)


def validate_map(y_true: np.ndarray, scores: np.ndarray, name: str) -> dict:
    """Evalue une carte de susceptibilite en la binarisant au seuil median."""
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    y_pred = (scores > np.median(scores)).astype(int)

    return {
        'modele':    name,
        'auc':       roc_auc_score(y_true, scores_norm),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'cm':        confusion_matrix(y_true, y_pred),
    }


def build_hybrid_maps(df: pd.DataFrame,
                      poids_ahp: dict[str, float],
                      poids_ml_cols: dict[str, float],
                      norm_mapping: dict[str, str],
                      alpha_expert: float = 0.70,
                      class_labels: dict[int, str] | None = None) -> pd.DataFrame:
    """
    Calcule trois cartes de susceptibilite dans df et les valide :
      - susceptibilite_ahp     (AHP classique)
      - susceptibilite_hybrid  (poids hybrides)
    La proba ML (proba_ml) doit deja etre dans df.

    Retourne le DataFrame enrichi.
    """
    if class_labels is None:
        class_labels = {0: 'Tres bas', 1: 'Bas', 2: 'Moyen', 3: 'Haut', 4: 'Tres haut'}

    poids_ml = ml_weights_to_ahp_names(poids_ml_cols)
    poids_hybrid = compute_hybrid_weights(poids_ahp, poids_ml, alpha_expert)

    directions = getattr(config, 'FACTOR_DIRECTIONS', None)
    df['susceptibilite_ahp']    = apply_ahp_weights(df, poids_ahp,    norm_mapping, directions)
    df['susceptibilite_hybrid'] = apply_ahp_weights(df, poids_hybrid, norm_mapping, directions)

    for col_base in ('susceptibilite_ahp', 'susceptibilite_hybrid'):
        col_class = col_base + '_class'
        col_label = col_base + '_label'
        df[col_class] = classify_susceptibility(df[col_base].values)
        df[col_label] = df[col_class].map(class_labels)

    return df, poids_hybrid


def print_weight_analysis(df_compare: pd.DataFrame) -> None:
    print("\n=== COMPARAISON POIDS AHP vs ML ===")
    print(df_compare.to_string(index=False))
    print("\nAnalyse des ecarts :")
    for _, row in df_compare.iterrows():
        ecart = row['Ecart (%)']
        if abs(ecart) < 3:
            status = "CONSENSUS"
        elif abs(ecart) < 7:
            status = "DIVERGENCE MODEREE"
        else:
            status = "DIVERGENCE IMPORTANTE"
        print(f"  {row['Critere']:12s} : {row['AHP Expert (%)']:5.1f}% -> {row['ML Data (%)']:5.1f}%"
              f"  ({ecart:+.1f}%)  [{status}]")


def save_weights_json(poids_ahp: dict, poids_ml: dict,
                      poids_final: dict, output_dir: Path) -> None:
    data = {
        'ahp':   {k: float(v) for k, v in poids_ahp.items()},
        'ml':    {k: float(v) for k, v in poids_ml.items()},
        'final': {k: float(v) for k, v in poids_final.items()},
    }
    path = output_dir / "poids_comparison.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Poids JSON sauvegardes : {path}")
