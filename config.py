from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

for d in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Emprise geographique reelle Pikine & Keur Massar (WGS84)
# Source : cartes facteurs Sy et al. UTM Zone 28N converties
# ---------------------------------------------------------------------------

# Polygone approximatif de la zone d'etude (lat, lon) sens horaire
# Derive visuellement des planches cartographiques des 6 facteurs
STUDY_POLYGON = [
    (14.727, -17.448),  # SW - extr. ouest (commune 13)
    (14.740, -17.445),  # W
    (14.752, -17.442),  # NW commune 1/8
    (14.762, -17.438),  # NW
    (14.773, -17.428),  # N-ouest commune 15
    (14.787, -17.405),  # N commune 15
    (14.800, -17.390),  # N commune 17 ouest
    (14.810, -17.365),  # N sommet commune 17
    (14.812, -17.340),  # N
    (14.808, -17.318),  # N commune 16 haut
    (14.800, -17.298),  # NE commune 16
    (14.790, -17.280),  # NE commune 18
    (14.778, -17.274),  # E commune 18
    (14.762, -17.274),  # E commune 19
    (14.748, -17.278),  # SE commune 2
    (14.730, -17.283),  # SE bas commune 2
    (14.726, -17.295),  # S
    (14.726, -17.330),  # S commune 4
    (14.726, -17.365),  # S commune 3
    (14.726, -17.395),  # S commune 5
    (14.726, -17.420),  # S commune 1
    (14.727, -17.448),  # retour SW
]

# Centres approx des 19 communes (lat, lon, num_commune)
COMMUNE_CENTERS = [
    (14.745, -17.435, "1"),
    (14.733, -17.300, "2"),
    (14.743, -17.358, "3"),
    (14.731, -17.367, "4"),
    (14.748, -17.375, "5"),
    (14.752, -17.382, "6"),
    (14.756, -17.395, "7"),
    (14.754, -17.408, "8"),
    (14.753, -17.392, "9"),
    (14.750, -17.386, "10"),
    (14.745, -17.398, "11"),
    (14.748, -17.415, "12"),
    (14.738, -17.432, "13"),
    (14.762, -17.362, "14"),
    (14.778, -17.370, "15"),
    (14.786, -17.313, "16"),
    (14.800, -17.350, "17"),
    (14.778, -17.280, "18"),
    (14.763, -17.300, "19"),
]

# Bornes encadrantes (bounding box)
LAT_MIN, LAT_MAX = 14.726, 14.813
LON_MIN, LON_MAX = -17.450, -17.272

# ---------------------------------------------------------------------------
# Parametres dataset synthetique
# ---------------------------------------------------------------------------
N_PIXELS     = 5000
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Parametres AHP
# ---------------------------------------------------------------------------
CRITERIA = ['pluie', 'altitude', 'pente', 'drainage', 'sol', 'usage']

SAATY_MATRIX_VALUES = {
    ('pluie', 'altitude'): 1.5,
    ('pluie', 'pente'):    2.0,
    ('pluie', 'drainage'): 1.7,
    ('pluie', 'sol'):      3.0,
    ('pluie', 'usage'):    6.0,
    ('altitude', 'pente'):    1.5,
    ('altitude', 'drainage'): 0.9,
    ('altitude', 'sol'):      2.0,
    ('altitude', 'usage'):    4.0,
    ('pente', 'drainage'): 0.6,
    ('pente', 'sol'):      1.2,
    ('pente', 'usage'):    2.5,
    ('drainage', 'sol'):   1.8,
    ('drainage', 'usage'): 3.5,
    ('sol', 'usage'):      2.0,
}

NORM_MAPPING = {
    'pluie':    'pluie_norm',
    'altitude': 'altitude_norm',
    'pente':    'pente_norm',
    'drainage': 'drainage_norm',
    'sol':      'sol_norm',
    'usage':    'usage_norm',
}

# +1 = valeur haute -> risque haut | -1 = valeur basse -> risque haut
FACTOR_DIRECTIONS = {
    'pluie':    +1,
    'altitude': -1,
    'pente':    -1,
    'drainage': -1,
    'sol':      +1,
    'usage':    +1,
}

# ---------------------------------------------------------------------------
# Parametres ML
# ---------------------------------------------------------------------------
TEST_SIZE    = 0.3
FEATURE_COLS = ['altitude_norm', 'pente_norm', 'pluie_norm',
                'drainage_norm', 'sol_norm', 'usage_norm']

ALPHA_EXPERT = 0.70
ALPHA_DATA   = 0.30

# ---------------------------------------------------------------------------
# Parametres LSTM
# ---------------------------------------------------------------------------
N_MONTHS         = 120
N_ZONES          = 100
LOOKBACK         = 6
LSTM_EPOCHS      = 50
LSTM_BATCH_SIZE  = 32

CLASS_LABELS = {0: 'Tres bas', 1: 'Bas', 2: 'Moyen', 3: 'Haut', 4: 'Tres haut'}

NORMALIZATION_BOUNDS = {
    'altitude': (0.0,  21.5),
    'pente':    (0.0,  21.7),
    'pluie':    (67.0, 95.0),
    'drainage': (0.0,   4.6),
    'sol':      (0,     4),
    'usage':    (0,     6),
}
