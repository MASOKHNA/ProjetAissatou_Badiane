"""
Visualisations et rapport pour le projet AHP+IA Pikine.
Genere graphiques PNG, carte interactive Folium et rapport texte.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, confusion_matrix

import config

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'


# ---------------------------------------------------------------------------
# Poids
# ---------------------------------------------------------------------------

def plot_weights_comparison(poids_ahp: dict, poids_ml: dict,
                             poids_final: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    criteria = list(poids_ahp.keys())
    x = np.arange(len(criteria))
    w = 0.35

    ax = axes[0]
    b1 = ax.bar(x - w / 2, [poids_ahp[c] * 100 for c in criteria],
                w, label='AHP Expert', color='steelblue', alpha=0.85)
    b2 = ax.bar(x + w / 2, [poids_ml[c] * 100 for c in criteria],
                w, label='ML Data-Driven', color='coral', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, rotation=40, ha='right')
    ax.set_ylabel('Poids (%)')
    ax.set_title('AHP Expert vs ML Data-Driven')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    vals = [poids_final[c] * 100 for c in criteria]
    bars = ax.bar(criteria, vals, color='mediumpurple', alpha=0.85)
    ax.set_xticklabels(criteria, rotation=40, ha='right')
    ax.set_ylabel('Poids (%)')
    ax.set_title('Poids Finaux Hybrides (70% Expert + 30% Data)')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = output_dir / "poids_comparison.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sauvegarde : {path}")


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(df: pd.DataFrame,
                    metrics_ahp: dict, metrics_hybrid: dict, metrics_ml: dict,
                    output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    pairs = [
        ('susceptibilite_ahp',    'AHP',    'steelblue',    metrics_ahp),
        ('susceptibilite_hybrid', 'Hybrid', 'coral',        metrics_hybrid),
        ('proba_ml',              'ML Pur', 'mediumpurple', metrics_ml),
    ]
    for col, label, color, m in pairs:
        fpr, tpr, _ = roc_curve(df['inonde'], df[col] / df[col].max())
        ax.plot(fpr, tpr, label=f'{label} (AUC={m["auc"]:.3f})',
                color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatoire')
    ax.set_xlabel('Taux Faux Positifs')
    ax.set_ylabel('Taux Vrais Positifs')
    ax.set_title('Courbes ROC - Comparaison Modeles')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / "roc_curves.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sauvegarde : {path}")


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    y_true = df['inonde'].values
    configs = [
        ('susceptibilite_ahp',    'AHP Classique'),
        ('susceptibilite_hybrid', 'Hybrid AHP+ML'),
        ('proba_ml',              'ML Pur'),
    ]
    for ax, (col, title) in zip(axes, configs):
        scores = df[col].values
        y_pred = (scores > np.median(scores)).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Non-inonde', 'Inonde'])
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    path = output_dir / "confusion_matrices.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sauvegarde : {path}")


# ---------------------------------------------------------------------------
# LSTM training curves
# ---------------------------------------------------------------------------

def plot_lstm_training(history, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('LSTM - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    auc_key = next(
        (k for k in history.history if 'auc' in k.lower() and not k.lower().startswith('val')),
        None
    )
    val_key = next(
        (k for k in history.history if 'auc' in k.lower() and k.lower().startswith('val')),
        None
    )
    if auc_key and val_key:
        axes[1].plot(history.history[auc_key], label='Train AUC')
        axes[1].plot(history.history[val_key], label='Val AUC')
    else:
        axes[1].text(0.5, 0.5, 'AUC non disponible', ha='center', va='center',
                     transform=axes[1].transAxes)
    axes[1].set_title('LSTM - AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "lstm_training.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sauvegarde : {path}")


# ---------------------------------------------------------------------------
# Carte interactive Folium — Choroplèthe masquee sur la zone d'etude
# ---------------------------------------------------------------------------

# Palette continue vert → jaune → orange → rouge (RdYlGn inverse)
_GRADIENT = [
    (0.00, (26,  152,  80)),   # vert fonce
    (0.25, (145, 207,  96)),   # vert clair
    (0.50, (254, 224, 139)),   # jaune
    (0.75, (252, 141,  89)),   # orange
    (1.00, (215,  48,  39)),   # rouge
]


def _score_to_hex(score: float, max_score: float = 255.0) -> str:
    """Convertit un score [0, max_score] en couleur hex interpolee."""
    t = float(np.clip(score / max_score, 0, 1))
    for i in range(len(_GRADIENT) - 1):
        t0, c0 = _GRADIENT[i]
        t1, c1 = _GRADIENT[i + 1]
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            r = int(c0[0] + u * (c1[0] - c0[0]))
            g = int(c0[1] + u * (c1[1] - c0[1]))
            b = int(c0[2] + u * (c1[2] - c0[2]))
            return f'#{r:02x}{g:02x}{b:02x}'
    return '#d73027'


def _point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """
    Test point-dans-polygone par ray casting.
    polygon = [(lat, lon), ...] — sens quelconque.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _cells_in_polygon(lat_centers, lon_centers, polygon):
    """Retourne un tableau bool indiquant si chaque cellule est dans le polygone."""
    return np.array([
        _point_in_polygon(float(la), float(lo), polygon)
        for la, lo in zip(lat_centers, lon_centers)
    ])


def _class_label(s: float) -> str:
    if s < 51:  return 'Tres bas'
    if s < 102: return 'Bas'
    if s < 153: return 'Moyen'
    if s < 204: return 'Haut'
    return 'Tres haut'


def build_folium_map(df: pd.DataFrame, output_dir: Path,
                     n_rows: int = 35, n_cols: int = 65) -> None:
    """
    Carte choroplèthe en grille masquee sur le polygone Pikine/Keur Massar.
    Seules les cellules dont le centre est a l'interieur du polygone d'etude
    sont colorees — reproduit le style des planches cartographiques de reference.
    """
    polygon  = config.STUDY_POLYGON
    lat_min  = config.LAT_MIN
    lat_max  = config.LAT_MAX
    lon_min  = config.LON_MIN
    lon_max  = config.LON_MAX
    bounds   = [[lat_min, lon_min], [lat_max, lon_max]]
    center   = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols

    # Precomputer les centres de cellule et masque polygone
    row_idx = np.arange(n_rows)
    col_idx = np.arange(n_cols)
    row_g, col_g = np.meshgrid(row_idx, col_idx, indexing='ij')
    row_g = row_g.ravel()
    col_g = col_g.ravel()
    lat_c = lat_min + (row_g + 0.5) * lat_step
    lon_c = lon_min + (col_g + 0.5) * lon_step
    in_poly = _cells_in_polygon(lat_c, lon_c, polygon)

    # Affecter chaque point du dataset a sa cellule
    df2 = df.copy()
    df2['_row'] = ((df2['lat'] - lat_min) / lat_step).astype(int).clip(0, n_rows - 1)
    df2['_col'] = ((df2['lon'] - lon_min) / lon_step).astype(int).clip(0, n_cols - 1)
    df2['_cell'] = df2['_row'] * n_cols + df2['_col']

    grid = df2.groupby('_cell').agg(
        suscept_hybrid=('susceptibilite_hybrid', 'mean'),
        suscept_ahp   =('susceptibilite_ahp',    'mean'),
        proba_ml      =('proba_ml',              'mean'),
        inonde_pct    =('inonde',                'mean'),
        n             =('susceptibilite_hybrid', 'count'),
        altitude      =('altitude_m',            'mean'),
        pluie         =('pluie_mm',              'mean'),
        drainage      =('drainage_km_km2',       'mean'),
    ).reset_index()

    # Creer la carte Folium
    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles='CartoDB positron',
        max_bounds=True,
        min_zoom=11,
        max_zoom=17,
    )
    m.fit_bounds(bounds)

    # ----- Couche 1 : Susceptibilite Hybride -----
    layer_hybrid = folium.FeatureGroup(name='Susceptibilite Hybride (AHP+ML)', show=True)
    for cell_id, in_p, la, lo in zip(
        range(n_rows * n_cols), in_poly, lat_c, lon_c
    ):
        if not in_p:
            continue
        row = grid[grid['_cell'] == cell_id]
        if row.empty:
            continue
        r = row.iloc[0]
        lat1, lon1 = la - lat_step / 2, lo - lon_step / 2
        lat2, lon2 = la + lat_step / 2, lo + lon_step / 2
        color = _score_to_hex(r['suscept_hybrid'])
        label = _class_label(r['suscept_hybrid'])
        commune_nom = 'Keur Massar' if lo > -17.37 else 'Pikine'
        tip = (
            f"<b>{label}</b> — {commune_nom}<br>"
            f"Score Hybride : {r['suscept_hybrid']:.0f}/255<br>"
            f"Score AHP    : {r['suscept_ahp']:.0f}/255<br>"
            f"Proba ML     : {r['proba_ml']:.0%}<br>"
            f"Taux inonde  : {r['inonde_pct']:.0%}<br>"
            f"Altitude moy : {r['altitude']:.1f} m<br>"
            f"Pluie moy    : {r['pluie']:.1f} mm<br>"
            f"Drainage     : {r['drainage']:.2f} km/km²<br>"
            f"N pixels     : {int(r['n'])}"
        )
        folium.Rectangle(
            bounds=[[lat1, lon1], [lat2, lon2]],
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            color=color,
            weight=0,
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(layer_hybrid)
    layer_hybrid.add_to(m)

    # ----- Couche 2 : Probabilite ML -----
    layer_ml = folium.FeatureGroup(name='Probabilite ML (Ensemble)', show=False)
    for cell_id, in_p, la, lo in zip(
        range(n_rows * n_cols), in_poly, lat_c, lon_c
    ):
        if not in_p:
            continue
        row = grid[grid['_cell'] == cell_id]
        if row.empty:
            continue
        r = row.iloc[0]
        lat1, lon1 = la - lat_step / 2, lo - lon_step / 2
        lat2, lon2 = la + lat_step / 2, lo + lon_step / 2
        color = _score_to_hex(r['proba_ml'] * 255)
        tip = (
            f"Proba ML : {r['proba_ml']:.0%}<br>"
            f"Taux inondation : {r['inonde_pct']:.0%}"
        )
        folium.Rectangle(
            bounds=[[lat1, lon1], [lat2, lon2]],
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            color=color,
            weight=0,
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(layer_ml)
    layer_ml.add_to(m)

    # ----- Couche 3 : Zones inondees observees -----
    layer_obs = folium.FeatureGroup(name='Zones inondees observees', show=False)
    for cell_id, in_p, la, lo in zip(
        range(n_rows * n_cols), in_poly, lat_c, lon_c
    ):
        if not in_p:
            continue
        row = grid[grid['_cell'] == cell_id]
        if row.empty or row.iloc[0]['inonde_pct'] < 0.30:
            continue
        r = row.iloc[0]
        lat1, lon1 = la - lat_step / 2, lo - lon_step / 2
        lat2, lon2 = la + lat_step / 2, lo + lon_step / 2
        alpha = min(0.90, 0.30 + r['inonde_pct'] * 0.60)
        folium.Rectangle(
            bounds=[[lat1, lon1], [lat2, lon2]],
            fill=True,
            fill_color='#1a73e8',
            fill_opacity=alpha,
            color='#1a73e8',
            weight=0,
            tooltip=folium.Tooltip(f"Taux inondation : {r['inonde_pct']:.0%}", sticky=True),
        ).add_to(layer_obs)
    layer_obs.add_to(m)

    # ----- Contour du polygone d'etude -----
    folium.Polygon(
        locations=polygon,
        color='#1a1a1a',
        fill=False,
        weight=2.0,
        opacity=0.9,
        tooltip="Zone d'etude : Departement de Pikine",
    ).add_to(m)

    # ----- Etiquettes des communes (numerotation) -----
    layer_communes = folium.FeatureGroup(name='Numeros communes', show=True)
    for lat_c_num, lon_c_num, num in config.COMMUNE_CENTERS:
        folium.Marker(
            location=[lat_c_num, lon_c_num],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:9px;font-weight:bold;color:#333;'
                    f'background:rgba(255,255,255,0.70);padding:1px 3px;'
                    f'border-radius:2px;border:1px solid #888;">{num}</div>'
                ),
                icon_size=(20, 16),
                icon_anchor=(10, 8),
            ),
        ).add_to(layer_communes)
    layer_communes.add_to(m)

    # ----- Etiquettes zones principales -----
    for name, loc, size in [
        ('PIKINE',        [14.750, -17.405], '13px'),
        ('KEUR MASSAR',   [14.775, -17.310], '13px'),
    ]:
        folium.Marker(
            location=loc,
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:{size};font-weight:bold;color:#111;'
                    f'background:rgba(255,255,255,0.82);padding:3px 7px;'
                    f'border-radius:4px;letter-spacing:0.5px;'
                    f'border:1.5px solid #555;white-space:nowrap;">{name}</div>'
                ),
                icon_size=(140, 26),
                icon_anchor=(70, 13),
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    # ----- Legende gradient -----
    gradient_css = ''.join([
        f'#{r:02x}{g:02x}{b:02x} {int(t * 100)}%,'
        for t, (r, g, b) in _GRADIENT
    ]).rstrip(',')

    legend_html = f"""
    <div style="position:fixed;bottom:25px;left:15px;z-index:9999;
                background:rgba(255,255,255,0.96);padding:12px 16px;
                border:1.5px solid #aaa;border-radius:8px;
                font-family:Arial,sans-serif;font-size:12px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.22);min-width:195px;">
      <b style="font-size:13px;">Susceptibilite inondation</b><br>
      <span style="font-size:10px;color:#666;">Pikine &amp; Keur Massar — Modele Hybride AHP+ML</span>
      <div style="margin:8px 0 2px;">
        <div style="height:15px;width:100%;border-radius:3px;
             background:linear-gradient(to right,{gradient_css});
             border:1px solid #ccc;"></div>
        <div style="display:flex;justify-content:space-between;
             font-size:9px;color:#444;margin-top:2px;">
          <span>Tres bas</span><span>Bas</span><span>Moyen</span><span>Haut</span><span>Tres haut</span>
        </div>
      </div>
      <hr style="margin:6px 0;border-color:#ddd;">
      <div style="font-size:10px;color:#555;line-height:1.5;">
        Survol cellule = details<br>
        Couches : panneau haut-droite<br>
        Projection : WGS 84
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ----- Titre cartographique -----
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                z-index:9999;background:rgba(255,255,255,0.95);
                padding:8px 18px;border-radius:6px;
                font-family:Arial,sans-serif;text-align:center;
                border:1.5px solid #aaa;box-shadow:1px 1px 6px rgba(0,0,0,0.15);">
      <b style="font-size:14px;color:#1a1a1a;">
        Carte de susceptibilite aux inondations
      </b><br>
      <span style="font-size:11px;color:#555;">
        Departement de Pikine &amp; Keur Massar — Fusion AHP + ML + LSTM
      </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    path = output_dir / "carte_pikine_hybrid.html"
    m.save(str(path))
    print(f"  Sauvegarde : {path}")


# ---------------------------------------------------------------------------
# Rapport texte
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame,
                    ahp_cr: float,
                    poids_ahp: dict, poids_ml: dict, poids_final: dict,
                    df_compare: pd.DataFrame,
                    metrics_ahp: dict, metrics_hybrid: dict, metrics_ml: dict,
                    auc_lstm: float, f1_lstm: float,
                    ml_summary: pd.DataFrame,
                    output_dir: Path) -> str:

    best_row = ml_summary.loc[ml_summary['AUC-ROC'].idxmax()]

    lines = [
        "=" * 70,
        "POC TECHNIQUE : ADAPTATION AHP AVEC INTELLIGENCE ARTIFICIELLE",
        "CARTOGRAPHIE SUSCEPTIBILITE INONDATION - PIKINE & KEUR MASSAR",
        "=" * 70,
        "",
        "1. RESUME EXECUTIF",
        "-" * 70,
        "Adaptation de la methodologie AHP classique (Sy et al., 2025) par",
        "integration de Machine Learning et Deep Learning (LSTM) pour ameliorer",
        "la cartographie des zones inondables de Pikine et Keur Massar.",
        "",
        "2. DONNEES",
        "-" * 70,
        f"  Dataset synthetique Pikine : {df.shape[0]} pixels x {df.shape[1]} colonnes",
        f"  Taux inondation observe   : {df['inonde'].mean():.1%}",
        f"  Facteurs AHP : altitude, pente, pluie, drainage, sol, usage",
        f"  Distributions spatiales calibrees sur planches cartographiques reelles",
        "",
        "3. POIDS AHP (Saaty)",
        "-" * 70,
    ]
    for crit, w in sorted(poids_ahp.items(), key=lambda x: -x[1]):
        lines.append(f"  {crit:12s} : {w*100:5.2f}%")
    status = "ACCEPTABLE" if ahp_cr < 0.10 else "A REVISER"
    lines += [f"\n  Consistency Ratio : {ahp_cr:.4f}  [{status}]", ""]

    lines += ["4. POIDS ML (moyenne RF + XGBoost + LightGBM)", "-" * 70]
    for crit, w in sorted(poids_ml.items(), key=lambda x: -x[1]):
        lines.append(f"  {crit:12s} : {w*100:5.2f}%")
    lines.append("")

    lines += ["5. COMPARAISON AHP vs ML", "-" * 70,
              df_compare.to_string(index=False), ""]

    lines += ["6. POIDS FINAUX HYBRIDES (70% Expert + 30% Data)", "-" * 70]
    for crit, w in sorted(poids_final.items(), key=lambda x: -x[1]):
        lines.append(f"  {crit:12s} : {w*100:5.2f}%")
    lines.append("")

    lines += ["7. VALIDATION CARTES", "-" * 70]
    for met, name in [(metrics_ahp, 'AHP Classique'),
                      (metrics_hybrid, 'Hybrid AHP+ML'),
                      (metrics_ml, 'ML Pur (Ensemble)')]:
        lines.append(f"  {name:22s} | AUC={met['auc']:.3f} | F1={met['f1']:.3f}"
                     f" | Prec={met['precision']:.3f} | Rec={met['recall']:.3f}")
    lines.append("")

    imp = (metrics_hybrid['auc'] - metrics_ahp['auc']) * 100
    lines += [f"  Amelioration AHP -> Hybrid : {imp:+.1f}% AUC", ""]

    lines += ["8. PERFORMANCES ML", "-" * 70, ml_summary.to_string(index=False),
              f"\n  Meilleur modele : {best_row['Modele']} (AUC={best_row['AUC-ROC']:.3f})", ""]

    lines += ["9. LSTM TEMPOREL", "-" * 70,
              f"  AUC-ROC : {auc_lstm:.3f}",
              f"  F1-Score : {f1_lstm:.3f}",
              "  Architecture : LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(1)",
              "  Lookback : 6 mois", ""]

    lines += ["10. PROCHAINES ETAPES", "-" * 70,
              "  PHASE 1 : Collecter GeoTIFF Sentinel-2 / SRTM (Google Earth Engine)",
              "  PHASE 2 : Compiler historique inondations 2015-2024 (Mairie Pikine/ONAS)",
              "  PHASE 3 : Reentraine modeles sur donnees reelles",
              "  PHASE 4 : Deploiement API FastAPI + dashboard Streamlit + alertes SMS",
              "",
              "=" * 70,
              f"Rapport genere : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
              "Status POC : COMPLET",
              "=" * 70]

    report = "\n".join(lines)
    path = output_dir / "rapport_poc.txt"
    path.write_text(report, encoding='utf-8')
    print(f"  Rapport sauvegarde : {path}")
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_visuals(df, poids_ahp, poids_ml, poids_final, df_compare,
                         metrics_ahp, metrics_hybrid, metrics_ml,
                         lstm_results, ml_summary, ahp_cr, output_dir: Path) -> None:
    print("\n=== GENERATION VISUALISATIONS ===")
    output_dir.mkdir(exist_ok=True)
    plot_weights_comparison(poids_ahp, poids_ml, poids_final, output_dir)
    plot_roc_curves(df, metrics_ahp, metrics_hybrid, metrics_ml, output_dir)
    plot_confusion_matrices(df, output_dir)
    if lstm_results:
        plot_lstm_training(lstm_results['history'], output_dir)
    build_folium_map(df, output_dir)
    generate_report(
        df, ahp_cr, poids_ahp, poids_ml, poids_final, df_compare,
        metrics_ahp, metrics_hybrid, metrics_ml,
        lstm_results['auc'] if lstm_results else 0.0,
        lstm_results['f1']  if lstm_results else 0.0,
        ml_summary, output_dir,
    )
    print("Toutes les visualisations generees.")
