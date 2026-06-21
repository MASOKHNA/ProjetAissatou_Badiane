"""
Implementation de la methode AHP (Analytic Hierarchy Process) - Saaty.
Calcule les poids des criteres d'inondation et verifie la coherence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Indice Aleatoire de Saaty selon le nombre de criteres
_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
       6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.51}


class AHPWeights:
    """
    Calcule les poids AHP via comparaisons par paires (matrice de Saaty).

    Usage :
        ahp = AHPWeights(['pluie', 'altitude', 'pente', 'drainage', 'sol', 'usage'])
        ahp.set_saaty_matrix({('pluie', 'altitude'): 1.5, ...})
        weights = ahp.calculate_weights()
    """

    def __init__(self, criteria: list[str]):
        self.criteria = criteria
        self.n = len(criteria)
        self.matrix: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.consistency_ratio: float | None = None

    def set_saaty_matrix(self, values: dict[tuple[str, str], float]) -> None:
        """
        Construit la matrice de comparaisons par paires n x n.

        values : dict dont les cles sont (critere_i, critere_j) et les valeurs
                 sont les scores Saaty 1-9. La diagonale et les inverses sont
                 remplis automatiquement.
        """
        matrix = np.ones((self.n, self.n))

        for (ci, cj), val in values.items():
            i = self.criteria.index(ci)
            j = self.criteria.index(cj)
            matrix[i, j] = val
            matrix[j, i] = 1.0 / val

        self.matrix = matrix

    def calculate_weights(self) -> np.ndarray:
        """
        Derive les poids AHP par normalisation des colonnes (approximation
        de la methode eigenvalue). Calcule aussi le Consistency Ratio (CR).

        Retourne un tableau numpy de poids (somme = 1).
        Leve ValueError si la matrice n'a pas ete definie.
        """
        if self.matrix is None:
            raise ValueError("Matrice Saaty non definie. Appeler set_saaty_matrix() d'abord.")

        col_sums = self.matrix.sum(axis=0)
        normalized = self.matrix / col_sums
        weights = normalized.mean(axis=1)
        weights /= weights.sum()
        self.weights = weights

        self._compute_cr()
        return weights

    def _compute_cr(self) -> None:
        """Calcule le Consistency Index (CI) et le Consistency Ratio (CR)."""
        lam_max = (self.matrix @ self.weights / self.weights).mean()
        ci = (lam_max - self.n) / (self.n - 1)
        ri = _RI.get(self.n, 1.51)
        self.consistency_ratio = ci / ri if ri > 0 else 0.0

    def weights_dict(self) -> dict[str, float]:
        if self.weights is None:
            raise ValueError("Poids non calcules. Appeler calculate_weights() d'abord.")
        return dict(zip(self.criteria, self.weights))

    def summary(self) -> pd.DataFrame:
        """Retourne un DataFrame trie des poids AHP (%)."""
        if self.weights is None:
            raise ValueError("Poids non calcules.")
        df = pd.DataFrame({
            'Critere': self.criteria,
            'Poids (%)': self.weights * 100,
        }).sort_values('Poids (%)', ascending=False).reset_index(drop=True)
        return df

    def print_summary(self) -> None:
        print("\n=== POIDS AHP (methode Saaty) ===")
        print(self.summary().to_string(index=False))
        cr = self.consistency_ratio
        status = "ACCEPTABLE" if cr < 0.10 else "A REVISER"
        print(f"\nConsistency Ratio : {cr:.4f}  [{status}]")


def apply_ahp_weights(df: pd.DataFrame,
                      weights: dict[str, float],
                      norm_mapping: dict[str, str],
                      directions: dict[str, int] | None = None) -> np.ndarray:
    """
    Calcule la susceptibilite AHP pour chaque ligne du DataFrame.

    weights      : dict critere -> poids (somme = 1)
    norm_mapping : dict critere -> nom de colonne normalisee dans df (0-255)
    directions   : dict critere -> +1 (valeur haute = risque haut)
                                or -1 (valeur basse = risque haut, ex: altitude, pente)
                   Si None, toutes les directions sont +1 (comportement original).

    Retourne un tableau numpy de valeurs [0, 255].
    """
    result = np.zeros(len(df))
    for critere, w in weights.items():
        col = norm_mapping[critere]
        vals = df[col].values
        if directions is not None and directions.get(critere, +1) == -1:
            vals = 255.0 - vals
        result += w * vals
    return result


def classify_susceptibility(values: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """
    Classe les valeurs de susceptibilite en n_classes par quantiles egaux.
    Retourne des entiers 0..n_classes-1.
    """
    bins = np.percentile(values, np.linspace(0, 100, n_classes + 1))
    classes = np.digitize(values, bins) - 1
    return np.clip(classes, 0, n_classes - 1)
