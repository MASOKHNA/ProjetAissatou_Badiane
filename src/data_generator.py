"""
Generation de donnees synthetiques simulant Pikine & Keur Massar (Senegal).
Distributions spatiales calibrees sur les 6 planches cartographiques reelles :
  - Altitude   : 0-21.5 m   (plat dominant, dunes au nord)
  - Pente      : 0-21.7 %   (quasi-plat, quelques pentes sur dunes)
  - Pluie      : 67-95 mm   (gradient ouest->est)
  - Drainage   : 0-4.6 km2  (faible dominant, taches dispersees)
  - Sol        : 0-4        (hydromorphes dominant, dunes nord, ferreux est)
  - Usage      : 0-5        (culture maraichere dominant, localites nord)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# Limites de la zone d'etude
_LAT_MIN, _LAT_MAX = 14.726, 14.813
_LON_MIN, _LON_MAX = -17.450, -17.272

# Codes et risques des types de sol (sens positif : valeur haute = risque haut)
# 0=Dunes Littorales, 1=Ferrugineux tropicaux, 2=Halomorphes,
# 3=Hydromorphes (dominant, risque max), 4=Eau
SOL_CODES = {0: 'Dunes', 1: 'Ferrugineux', 2: 'Halomorphes', 3: 'Hydromorphes', 4: 'Eau'}

# Codes usage du sol
# 0=Savane/Steppe, 1=Culture maraichere, 2=Lac, 3=Mare, 4=Savane arboree, 5=Localite
USAGE_CODES = {0: 'Savane', 1: 'Culture maraichere', 2: 'Lac',
               3: 'Mare', 4: 'Savane arboree', 5: 'Localite'}


def _norm_pos(lat, lon):
    """Retourne (lat_norm [0,1], lon_norm [0,1]) dans la bounding box."""
    ln = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)
    rn = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN)
    return np.clip(ln, 0, 1), np.clip(rn, 0, 1)


def _gen_coords(n, rng):
    """Tire des coordonnees uniformes dans la bounding box."""
    lat = rng.uniform(_LAT_MIN, _LAT_MAX, n)
    lon = rng.uniform(_LON_MIN, _LON_MAX, n)
    return lat, lon


def _altitude(lat_n, lon_n, n, rng):
    """
    Altitude reelle Pikine d'apres la planche cartographique :
    - Zone basse (0-3 m) : majorite de la zone, surtout centre-sud
    - Dunes nord (communes 15-17) : 8-21 m — lat_n > 0.72
    - Quelques buttes dune ouest (communes 12-13) : 4-10 m — lon_n < 0.15
    - Sud-est plat (commune 2) : 0-2 m — lat_n < 0.18
    """
    base = rng.gamma(1.5, 1.5, n)                          # 0-8 m, median ~2 m

    # Crete dunaire nord
    dune_n = np.clip((lat_n - 0.72) / 0.28, 0, 1) ** 1.3
    dune_h = rng.gamma(4, 2.5, n)                          # 0-18 m sur dunes
    dune_n_effect = dune_n * dune_h * 14

    # Buttes dunaires ouest (communes 12-13)
    dune_w = np.clip((0.15 - lon_n) / 0.15, 0, 1) * np.clip((lat_n - 0.25) / 0.5, 0, 1)
    dune_w_effect = dune_w * rng.exponential(3.0, n)

    # Plaine alluviale sud (bas-fonds communes 2, 4)
    south_flat = np.clip((0.18 - lat_n) / 0.18, 0, 1) * rng.uniform(0, 1, n)

    alt = base + dune_n_effect + dune_w_effect - south_flat
    noise = rng.normal(0, 0.4, n)
    return np.clip(alt + noise, 0.0, 21.5)


def _pente(altitude, rng):
    """
    Pente presque nulle sur tout le domaine (0-0.85 % dominant).
    Les dunes (altitude > 8 m) ont des pentes plus marquees.
    """
    base = rng.exponential(0.25, len(altitude))
    dune_slope = np.where(altitude > 8, rng.exponential(1.5, len(altitude)), 0)
    slope = base + dune_slope * (altitude - 8).clip(0) / 13.5
    return np.clip(slope + rng.normal(0, 0.05, len(altitude)), 0.0, 21.7)


def _pluie(lon_n, n, rng):
    """
    Intensite pluviometrique (mm) — gradient ouest -> est observe sur la carte.
    Ouest : 67-72 mm  |  Centre : 77-82 mm  |  Est (Keur Massar) : 87-95 mm
    """
    mean_rain = 67.0 + lon_n * 28.0          # 67 mm (W) -> 95 mm (E)
    noise = rng.normal(0, 2.5, n)
    return np.clip(mean_rain + noise, 67.0, 94.82)


def _drainage(lat_n, lon_n, n, rng):
    """
    Densite drainage (km/km2) — faible dominant (0-0.44),
    taches dispersees de haute densite observees a l'ouest et au centre.
    """
    base = rng.exponential(0.3, n)                          # surtout 0-0.44

    # Taches haute densite (rouge sur la carte) — position aleatoire
    patch_seed = rng.uniform(0, 1, n)
    high_drain = np.where(patch_seed < 0.08, rng.uniform(2.0, 4.6, n), 0)

    # Zone ouest (communes 1, 7-8, 12) : drainage un peu plus eleve
    west_bonus = np.clip((0.25 - lon_n) / 0.25, 0, 1) * rng.exponential(0.5, n)

    drainage = base + high_drain + west_bonus
    return np.clip(drainage, 0.0, 4.635)


def _sol(lat_n, lon_n, n, rng):
    """
    Type de sol base sur la carte pedologique :
    - Dunes Littorales (0) : bande nord lat_n > 0.78, lon_n in [0.05, 0.72]
    - Ferrugineux tropicaux (1) : zone est, lon_n > 0.65
    - Halomorphes (2) : patches centre
    - Hydromorphes (3) : dominant centre-sud-ouest (>50 %)
    - Eau (4) : petits plans d'eau (lac nord, retenues)
    """
    sol = np.full(n, 3, dtype=int)    # Hydromorphes par defaut

    # Dunes litterales nord
    mask_dune = (lat_n > 0.78) & (lon_n > 0.05) & (lon_n < 0.72)
    sol[mask_dune] = 0

    # Ferrugineux tropicaux - est (Keur Massar, communes 16/18/19)
    mask_ferr = (lon_n > 0.65) & (lat_n < 0.78)
    sol[mask_ferr] = 1

    # Halomorphes - patches centre (communes 3, 5, 6, 9, 14)
    halomorph_lon = (lon_n > 0.25) & (lon_n < 0.60)
    halomorph_lat = (lat_n > 0.20) & (lat_n < 0.65)
    mask_halo = halomorph_lon & halomorph_lat & (rng.uniform(0, 1, n) < 0.35)
    sol[mask_halo & (sol == 3)] = 2   # remplace que des Hydromorphes

    # Eau - lac nord (commune 17) + mares dispersees
    mask_eau_nord = (lat_n > 0.82) & (lon_n > 0.40) & (lon_n < 0.60)
    sol[mask_eau_nord] = 4
    mask_eau_mare = rng.uniform(0, 1, n) < 0.025
    sol[mask_eau_mare] = 4

    return sol


def _usage(lat_n, lon_n, sol, n, rng):
    """
    Occupation du sol based on reference map :
    - Localite/Urbain (5)        : nord dense (lat_n > 0.80) + zones agglomerees
    - Culture maraichere (1)     : dominant partout (Niayes, ~55 %)
    - Mare (3)                   : bas-fonds sud, zones humides
    - Lac (2)                    : plans d'eau
    - Savane arboree (4)         : est (Keur Massar)
    - Savane/Steppe (0)          : quelques zones est-sud
    """
    usage = np.full(n, 1, dtype=int)    # Culture maraichere par defaut

    # Localites au nord (communes 17-dense, 15 partiellement)
    mask_urb = lat_n > 0.80
    usage[mask_urb] = 5

    # Zones urbanisees dispersees (communes 7-11, Pikine centre)
    mask_urb2 = ((lat_n > 0.22) & (lat_n < 0.50) &
                 (lon_n > 0.15) & (lon_n < 0.55) &
                 (rng.uniform(0, 1, n) < 0.30))
    usage[mask_urb2] = 5

    # Mares et bas-fonds (sud commune 2, zones hydromorphes)
    mask_mare = (sol == 4) | ((lat_n < 0.20) & (rng.uniform(0, 1, n) < 0.40))
    usage[mask_mare] = 3

    # Lacs
    usage[sol == 4] = 2

    # Savane arboree - est
    mask_sav = (lon_n > 0.70) & (lat_n < 0.60) & (usage == 1)
    usage[mask_sav & (rng.uniform(0, 1, n) < 0.45)] = 4

    # Savane/Steppe quelques zones est
    mask_steppe = (lon_n > 0.75) & (lat_n < 0.45) & (usage == 1)
    usage[mask_steppe & (rng.uniform(0, 1, n) < 0.30)] = 0

    return usage


def _flood_label(altitude, pluie, drainage, sol, usage, lon_n, lat_n, n, rng):
    """
    Label d'inondation simule a partir des facteurs spatiaux.
    Calibre pour un taux d'inondation realiste ~40-55 %.
    Zones a haut risque : bas-fonds, sols hydromorphes, zones urbaines basses.
    """
    # Normaliser chaque facteur en [0,1] dans le sens du risque
    alt_risk  = 1.0 - np.clip(altitude / 21.5, 0, 1)     # bas = risque haut
    rain_risk = np.clip((pluie - 67) / 28, 0, 1)          # est = plus de pluie
    drn_risk  = 1.0 - np.clip(drainage / 4.635, 0, 1)     # faible drainage = risque
    sol_risk  = sol / 4.0                                   # hydromorphe / eau = max
    urb_risk  = np.where(usage == 5, 1.0,
                np.where(usage == 3, 0.7,
                np.where(usage == 2, 0.6, 0.2)))

    # Score composite (poids calques sur les poids AHP)
    score = (alt_risk  * 0.30 +
             rain_risk * 0.25 +
             drn_risk  * 0.20 +
             sol_risk  * 0.15 +
             urb_risk  * 0.10)

    # Zones bas-fonds sud : inondation quasi systematique
    score = np.where((lat_n < 0.18) & (sol_risk > 0.5), score * 1.20, score)

    # Threshold 0.64 centre sur la mediane empirique du score (~0.65)
    # => taux d'inondation realiste ~50-55 % (Pikine est tres vulnerable)
    prob = 1.0 / (1.0 + np.exp(-6 * (score - 0.64)))
    prob = np.clip(prob + rng.normal(0, 0.06, n), 0, 1)
    return (prob > 0.50).astype(int), np.clip(prob, 0, 1)


def create_synthetic_dataset(n_pixels: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Cree un dataset realiste simulant Pikine & Keur Massar avec distributions
    spatiales calibrees sur les 6 planches cartographiques de reference.

    Retourne un DataFrame avec :
    - 6 facteurs bruts (unites reelles) et leurs normalisees [0-255]
    - label binaire d'inondation et probabilite
    - coordonnees GPS et commune d'appartenance
    """
    rng = np.random.default_rng(random_state)

    lat, lon = _gen_coords(n_pixels, rng)
    lat_n, lon_n = _norm_pos(lat, lon)

    altitude = _altitude(lat_n, lon_n, n_pixels, rng)
    pente    = _pente(altitude, rng)
    pluie    = _pluie(lon_n, n_pixels, rng)
    drainage = _drainage(lat_n, lon_n, n_pixels, rng)
    sol      = _sol(lat_n, lon_n, n_pixels, rng)
    usage    = _usage(lat_n, lon_n, sol, n_pixels, rng)

    inonde, prob = _flood_label(altitude, pluie, drainage, sol, usage,
                                lon_n, lat_n, n_pixels, rng)

    # Normalisation 0-255 avec bornes reelles des cartes
    def _n255(arr, lo, hi):
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1) * 255.0

    alt_norm = _n255(altitude, 0.0, 21.5)
    pnt_norm = _n255(pente,    0.0, 21.7)
    plu_norm = _n255(pluie,   67.0, 94.82)
    drn_norm = _n255(drainage, 0.0,  4.635)
    sol_norm = _n255(sol.astype(float), 0.0, 4.0)
    usg_norm = _n255(usage.astype(float), 0.0, 5.0)

    commune = np.where(lon > -17.37, 'Keur Massar', 'Pikine')

    return pd.DataFrame({
        # Valeurs brutes
        'altitude_m':       altitude,
        'pente_pct':        pente,
        'pluie_mm':         pluie,
        'drainage_km_km2':  drainage,
        'sol_type':         sol,
        'usage_terre':      usage,
        # Normalisees 0-255 pour AHP
        'altitude_norm':    alt_norm,
        'pente_norm':       pnt_norm,
        'pluie_norm':       plu_norm,
        'drainage_norm':    drn_norm,
        'sol_norm':         sol_norm,
        'usage_norm':       usg_norm,
        # Labels
        'inonde':           inonde,
        'prob_inonde':      prob,
        # Geolocalisation
        'lat':              lat,
        'lon':              lon,
        'commune':          commune,
    })


def create_timeseries_dataset(n_zones: int = 100, n_months: int = 120) -> pd.DataFrame:
    """
    Dataset temporel (120 mois, 2015-2024) pour l'entrainement LSTM.
    Simule l'evolution mensuelle de facteurs hydrologiques sur n_zones.
    Saisonnalite marquee : saison des pluies juillet-septembre.
    """
    rng = np.random.default_rng(42)
    months = pd.date_range(start='2015-01', periods=n_months, freq='ME')
    records = []

    for zone_id in range(n_zones):
        # Chaque zone a ses caracteristiques de base
        pluie_base   = rng.uniform(67, 95)
        alt_base     = rng.uniform(0, 10)
        drainage_base = rng.uniform(0.1, 2.0)

        for idx, month in enumerate(months):
            # Saisonnalite centree sur aout (idx_saison maximal en juillet-aout)
            season = 0.5 + 0.5 * np.sin(2 * np.pi * (idx % 12 - 1) / 12)
            pluie  = pluie_base * (0.5 + 0.5 * season) + rng.normal(0, 3)
            drn    = drainage_base + rng.normal(0, 0.15)
            alt    = alt_base + rng.normal(0, 0.3)
            pente  = rng.exponential(0.3)
            sol    = int(rng.integers(0, 4))

            prob_flood = (
                (1 - np.clip(alt / 10, 0, 1)) * 0.30 +
                np.clip((pluie - 67) / 28, 0, 1) * 0.30 +
                (1 - np.clip(drn / 2.0, 0, 1)) * 0.25 +
                (sol / 4.0) * 0.15
            )
            inonde = int(prob_flood > rng.uniform(0.35, 0.65))

            records.append({
                'zone_id':   zone_id,
                'date':      month,
                'month_idx': idx,
                'pluie':     float(np.clip(pluie, 67, 95)),
                'altitude':  float(np.clip(alt, 0, 21.5)),
                'pente':     float(np.clip(pente, 0, 21.7)),
                'drainage':  float(np.clip(drn, 0, 4.635)),
                'sol':       float(sol),
                'inonde':    inonde,
            })

    return pd.DataFrame(records)


def load_real_geotiff(filepath_dict: dict) -> 'pd.DataFrame | None':
    """
    Charge des rasters GeoTIFF reels et les convertit en DataFrame pixelwise.
    Retourne None si rasterio n'est pas installe ou si aucun fichier n'est trouve.
    """
    if not _HAS_RASTERIO:
        print("  rasterio non installe (pip install rasterio)")
        return None

    arrays = {}
    for name, path in filepath_dict.items():
        try:
            with rasterio.open(path) as src:
                data = src.read(1).astype(float)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                arrays[name] = data
                print(f"  Charge : {name} {data.shape}")
        except Exception as e:
            print(f"  Ignore {name} : {e}")

    if not arrays:
        return None

    flat = {k: v.ravel() for k, v in arrays.items()}
    return pd.DataFrame(flat).dropna()
