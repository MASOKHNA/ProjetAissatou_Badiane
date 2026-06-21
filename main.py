"""
Point d'entree principal - POC AHP+IA Pikine & Keur Massar
Detecter les zones inondables par fusion AHP classique + Machine Learning + LSTM.

Usage :
    python main.py                  # pipeline complet
    python main.py --skip-lstm      # sauter le LSTM (plus rapide)
"""

import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import config
from src.data_generator import create_synthetic_dataset, create_timeseries_dataset
from src.ahp import AHPWeights
from src.ml_models import train_all_models
from src.hybrid import (ml_weights_to_ahp_names, compute_hybrid_weights,
                        compare_weights, validate_map, build_hybrid_maps,
                        print_weight_analysis, save_weights_json)
from src.lstm_model import train_lstm
from src.visualization import generate_all_visuals
from src.inference import predict_susceptibility


def parse_args():
    parser = argparse.ArgumentParser(description="POC AHP+IA Pikine")
    parser.add_argument('--skip-lstm', action='store_true',
                        help="Ne pas entrainer le LSTM (gain de temps)")
    parser.add_argument('--n-pixels', type=int, default=config.N_PIXELS,
                        help=f"Nombre de pixels synthetiques (defaut {config.N_PIXELS})")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # ETAPE 1 : Donnees synthetiques
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPE 1 : GENERATION DONNEES SYNTHETIQUES")
    print("=" * 60)

    df = create_synthetic_dataset(n_pixels=args.n_pixels,
                                  random_state=config.RANDOM_STATE)
    df.to_csv(config.DATA_DIR / "pikine_synthetic.csv", index=False)
    print(f"Dataset : {df.shape[0]} pixels x {df.shape[1]} colonnes")
    print(f"Taux inondation : {df['inonde'].mean():.1%}")

    # ------------------------------------------------------------------
    # ETAPE 2 : AHP classique
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPE 2 : AHP CLASSIQUE (SAATY)")
    print("=" * 60)

    ahp = AHPWeights(config.CRITERIA)
    ahp.set_saaty_matrix(config.SAATY_MATRIX_VALUES)
    ahp.calculate_weights()
    ahp.print_summary()

    poids_ahp = ahp.weights_dict()

    # ------------------------------------------------------------------
    # ETAPE 3 : Machine Learning
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPE 3 : ENTRAINEMENT MODELES ML")
    print("=" * 60)

    ml_results = train_all_models(
        df,
        feature_cols=config.FEATURE_COLS,
        models_dir=config.MODELS_DIR,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    ensemble   = ml_results['ensemble']
    scaler     = ml_results['scaler']
    X_scaled   = ml_results['X_scaled']
    ml_summary = ml_results['summary']

    # Probabilites ML sur tout le dataset
    df['proba_ml'] = ensemble.predict_proba(X_scaled)[:, 1]

    # ------------------------------------------------------------------
    # ETAPE 4 : Comparaison AHP vs ML + poids hybrides
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPE 4 : COMPARAISON AHP vs ML")
    print("=" * 60)

    poids_ml_cols = ml_results['poids_ml_cols']
    poids_ml      = ml_weights_to_ahp_names(poids_ml_cols)
    poids_hybrid  = compute_hybrid_weights(poids_ahp, poids_ml,
                                           config.ALPHA_EXPERT)

    df_compare = compare_weights(poids_ahp, poids_ml)
    print_weight_analysis(df_compare)

    save_weights_json(poids_ahp, poids_ml, poids_hybrid, config.OUTPUT_DIR)

    # ------------------------------------------------------------------
    # ETAPE 5 : Cartes hybrides et validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPE 5 : CARTES HYBRIDES & VALIDATION")
    print("=" * 60)

    df, _ = build_hybrid_maps(
        df, poids_ahp, poids_ml_cols, config.NORM_MAPPING,
        alpha_expert=config.ALPHA_EXPERT,
        class_labels=config.CLASS_LABELS,
    )

    y_true = df['inonde'].values
    metrics_ahp    = validate_map(y_true, df['susceptibilite_ahp'].values,    "AHP Classique")
    metrics_hybrid = validate_map(y_true, df['susceptibilite_hybrid'].values, "Hybrid AHP+ML")
    metrics_ml     = validate_map(y_true, df['proba_ml'].values,              "ML Pur")

    print(f"\n{'Modele':22s} | {'AUC':>6} | {'F1':>6} | {'Prec':>6} | {'Rec':>6}")
    print("-" * 60)
    for m in (metrics_ahp, metrics_hybrid, metrics_ml):
        print(f"{m['modele']:22s} | {m['auc']:6.3f} | {m['f1']:6.3f} | "
              f"{m['precision']:6.3f} | {m['recall']:6.3f}")

    # ------------------------------------------------------------------
    # ETAPE 6 : LSTM temporel (optionnel)
    # ------------------------------------------------------------------
    lstm_results = None
    if not args.skip_lstm:
        print("\n" + "=" * 60)
        print("ETAPE 6 : LSTM TEMPOREL")
        print("=" * 60)

        df_ts = create_timeseries_dataset(n_zones=config.N_ZONES,
                                          n_months=config.N_MONTHS)
        lstm_results = train_lstm(
            df_ts,
            lookback=config.LOOKBACK,
            epochs=config.LSTM_EPOCHS,
            batch_size=config.LSTM_BATCH_SIZE,
            models_dir=config.MODELS_DIR,
        )
    else:
        print("\n[LSTM ignore (--skip-lstm)]")

    # ------------------------------------------------------------------
    # ETAPE 7 : Visualisations & rapport
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPE 7 : VISUALISATIONS & RAPPORT")
    print("=" * 60)

    generate_all_visuals(
        df=df,
        poids_ahp=poids_ahp,
        poids_ml=poids_ml,
        poids_final=poids_hybrid,
        df_compare=df_compare,
        metrics_ahp=metrics_ahp,
        metrics_hybrid=metrics_hybrid,
        metrics_ml=metrics_ml,
        lstm_results=lstm_results,
        ml_summary=ml_summary,
        ahp_cr=ahp.consistency_ratio,
        output_dir=config.OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # ETAPE 8 : Exemple d'inference
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPE 8 : EXEMPLE INFERENCE")
    print("=" * 60)

    exemple = predict_susceptibility(
        altitude=4.0,   # Tres bas = risque eleve
        pente=0.3,      # Tres plat
        pluie=750.0,    # Pluie forte
        drainage=0.6,   # Drainage faible
        sol=0,          # Halomorphe (mauvais drainage)
        usage=1,        # Urbain dense
        poids_ahp=poids_ahp,
        poids_hybrid=poids_hybrid,
        ensemble_model=ensemble,
        scaler=scaler,
    )

    print("Zone test (Medina Fall - type inonde) :")
    print(f"  Score AHP    : {exemple['score_ahp']:.1f}  -> {exemple['classe_ahp']}")
    print(f"  Score Hybrid : {exemple['score_hybrid']:.1f}  -> {exemple['classe_hybrid']}")
    if 'proba_ml' in exemple:
        print(f"  Proba ML     : {exemple['proba_ml']:.3f}  -> {exemple['classe_ml']}")
    if exemple['alerte']:
        print("  *** ALERTE : Zone a haut risque d'inondation ***")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("POC TERMINE AVEC SUCCES")
    print(f"  Resultats dans : {config.OUTPUT_DIR}/")
    print(f"  Modeles dans   : {config.MODELS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
