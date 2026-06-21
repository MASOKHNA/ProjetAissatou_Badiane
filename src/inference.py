"""
Fonction d'inference pour predire la susceptibilite inondation
d'un pixel a partir de ses valeurs brutes.
Utilisable en production sans recharger le dataset d'entrainement.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


# Bornes de normalisation 0-255 calees sur les donnees reelles des planches carto
_BOUNDS = {
    'altitude': (0.0,  21.5),
    'pente':    (0.0,  21.7),
    'pluie':   (67.0,  94.82),
    'drainage': (0.0,   4.635),
    'sol':      (0.0,   4.0),
    'usage':    (0.0,   5.0),
}

_CLASS_THRESHOLDS = [64, 128, 192, 224]
_CLASS_LABELS     = ['Tres bas', 'Bas', 'Moyen', 'Haut', 'Tres haut']
_CLASS_COLORS     = ['green',    'lightgreen', 'orange', 'red', 'darkred']


def _norm255(value: float, low: float, high: float) -> float:
    """Normalise une valeur brute dans [0, 255]."""
    return float(np.clip((value - low) / (high - low + 1e-8), 0, 1) * 255)


def _classify(score: float) -> tuple[str, str]:
    for i, thr in enumerate(_CLASS_THRESHOLDS):
        if score < thr:
            return _CLASS_LABELS[i], _CLASS_COLORS[i]
    return _CLASS_LABELS[-1], _CLASS_COLORS[-1]


def predict_susceptibility(altitude: float,
                           pente: float,
                           pluie: float,
                           drainage: float,
                           sol: int,
                           usage: int,
                           poids_ahp: dict[str, float],
                           poids_hybrid: dict[str, float],
                           ensemble_model=None,
                           scaler=None) -> dict:
    """
    Predit la susceptibilite inondation pour un pixel.

    Parametres
    ----------
    altitude  : altitude en metres (0-20)
    pente     : pente en %        (0-2)
    pluie     : pluviometrie mm   (300-900)
    drainage  : densite drainage  (0.1-1.5)
    sol       : type sol          (0=Halomorphe, 1=Permeable, 2=Argileux)
    usage     : usage terres      (0=Eau, 1=Urbain, 2=Semi-urbain, 3=Agricole)
    poids_ahp    : dict critere -> poids AHP (somme=1)
    poids_hybrid : dict critere -> poids hybrides (somme=1)
    ensemble_model : modele sklearn .predict_proba (optionnel)
    scaler        : StandardScaler (optionnel, requis si ensemble_model fourni)

    Retourne un dict avec scores AHP, hybrid, ML (si dispo) et classe.
    """
    # Normalisation
    norms = {
        'altitude': _norm255(altitude, *_BOUNDS['altitude']),
        'pente':    _norm255(pente,    *_BOUNDS['pente']),
        'pluie':    _norm255(pluie,    *_BOUNDS['pluie']),
        'drainage': _norm255(drainage, *_BOUNDS['drainage']),
        'sol':      _norm255(sol,      *_BOUNDS['sol']),
        'usage':    _norm255(usage,    *_BOUNDS['usage']),
    }

    # Direction de susceptibilite : -1 = valeur basse -> risque haut
    _DIRECTIONS = {
        'pluie': +1, 'altitude': -1, 'pente': -1,
        'drainage': -1, 'sol': +1, 'usage': +1,
    }

    def _directed(crit: str, val: float) -> float:
        return (255.0 - val) if _DIRECTIONS.get(crit, +1) == -1 else val

    # Score AHP classique
    score_ahp = sum(poids_ahp[c] * _directed(c, norms[c]) for c in poids_ahp)

    # Score hybride
    score_hybrid = sum(poids_hybrid[c] * _directed(c, norms[c]) for c in poids_hybrid)

    label_ahp,    color_ahp    = _classify(score_ahp)
    label_hybrid, color_hybrid = _classify(score_hybrid)

    result = {
        'inputs': {
            'altitude': altitude, 'pente': pente, 'pluie': pluie,
            'drainage': drainage, 'sol': sol, 'usage': usage,
        },
        'score_ahp':       round(score_ahp,    2),
        'score_hybrid':    round(score_hybrid, 2),
        'classe_ahp':      label_ahp,
        'classe_hybrid':   label_hybrid,
        'couleur_ahp':     color_ahp,
        'couleur_hybrid':  color_hybrid,
        'alerte':          score_hybrid > 180,
    }

    # Prediction ML si modele disponible
    if ensemble_model is not None and scaler is not None:
        feat_order = ['altitude_norm', 'pente_norm', 'pluie_norm',
                      'drainage_norm', 'sol_norm', 'usage_norm']
        feat_map = {
            'altitude_norm': norms['altitude'], 'pente_norm': norms['pente'],
            'pluie_norm':    norms['pluie'],    'drainage_norm': norms['drainage'],
            'sol_norm':      norms['sol'],       'usage_norm': norms['usage'],
        }
        x_raw = np.array([[feat_map[f] for f in feat_order]])
        x_scaled = scaler.transform(x_raw)
        proba = float(ensemble_model.predict_proba(x_scaled)[0, 1])
        result['proba_ml'] = round(proba, 4)
        result['classe_ml'] = 'Inonde' if proba >= 0.5 else 'Non-inonde'

    return result


def load_models(models_dir: Path) -> dict:
    """Charge les artefacts sauvegardes depuis models_dir."""
    loaded = {}
    files = {
        'rf':       'rf_model.pkl',
        'xgb':      'xgb_model.pkl',
        'lgb':      'lgb_model.pkl',
        'ensemble': 'ensemble_model.pkl',
        'scaler':   'scaler.pkl',
    }
    for key, filename in files.items():
        path = models_dir / filename
        if path.exists():
            loaded[key] = pickle.load(open(path, 'rb'))
    return loaded


def batch_predict(records: list[dict],
                  poids_ahp: dict, poids_hybrid: dict,
                  ensemble_model=None, scaler=None) -> list[dict]:
    """Applique predict_susceptibility sur une liste de pixels."""
    return [
        predict_susceptibility(**rec,
                               poids_ahp=poids_ahp,
                               poids_hybrid=poids_hybrid,
                               ensemble_model=ensemble_model,
                               scaler=scaler)
        for rec in records
    ]
