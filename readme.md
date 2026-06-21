# Explication du projet : Détection des zones inondables par IA
## Pikine & Keur Massar — Fusion AHP + Machine Learning + LSTM

---

## Contexte et problème

Pikine et Keur Massar sont deux communes de la banlieue de Dakar (Sénégal) régulièrement frappées par des inondations catastrophiques, notamment pendant la saison des pluies (juillet–septembre). Des milliers de familles perdent leurs biens chaque année.

**La question posée :**
> *Peut-on cartographier avec précision les zones les plus vulnérables aux inondations, en combinant la méthode AHP classique des experts et l'intelligence artificielle ?*

---

## Les données de base : 6 facteurs de risque

Le projet repose sur **6 facteurs** issus de cartes géographiques réelles de Pikine (planches cartographiques Sy et al.) :

| Facteur | Ce qu'il mesure | Logique de risque |
|---|---|---|
| **Altitude** (m) | Hauteur du terrain (0 – 21,5 m) | Terrain bas → inondation plus facile |
| **Pente** (%) | Inclinaison du terrain (0 – 21,7 %) | Terrain plat → l'eau stagne |
| **Pluviométrie** (mm) | Quantité de pluie reçue (67 – 95 mm) | Plus de pluie → plus de risque |
| **Densité drainage** (km/km²) | Réseau d'évacuation des eaux (0 – 4,6) | Mauvais drainage → eau bloquée |
| **Type de sol** | Nature du sol (5 types) | Sol imperméable → risque élevé |
| **Occupation du sol** | Usage du terrain (6 types) | Zone urbaine → surface imperméable |

### Distributions spatiales réelles observées sur les cartes

```
ALTITUDE  : 0–3 m (80 % de la zone, très plat), dunes au Nord > 8 m
PENTE     : Quasi-nulle (0–0,85 % dominant), légères pentes sur dunes
PLUIE     : Gradient Ouest → Est : 67 mm (Pikine Ouest) → 94 mm (Keur Massar)
DRAINAGE  : Faible (0–0,44 km/km²) dominant, quelques taches à haute densité
SOL       : Hydromorphes dominant (50 %), Ferrugineux (Est), Dunes (Nord)
USAGE     : Culture maraîchère dominant (60 %), Localités (Nord urbain)
```

---

## Le pipeline en 8 étapes

```
  Données       AHP          ML           Fusion       LSTM        Carte
  spatiales  → Expert  →  (RF/XGB/LGB) → Hybride  → Temporel  → Interactive
    [1]         [2]          [3]           [4-5]       [6]          [7]
```

---

### Étape 1 — Génération des données synthétiques

Puisqu'on ne dispose pas encore de données terrain réelles numériques, on génère **5 000 pixels synthétiques** simulant les conditions géographiques réelles de Pikine.

Chaque pixel représente un carré du terrain avec ses 6 valeurs de facteurs, une coordonnée GPS et un label indiquant s'il est inondable ou non.

**Caractéristiques du dataset :**
- 5 000 pixels × 17 colonnes
- Taux d'inondation : **57,4 %** *(Pikine est très vulnérable, ce taux est réaliste)*
- Distributions calées sur les vraies cartes : gradient pluviométrique, zones dunaires, types de sol par zone

---

### Étape 2 — Méthode AHP (Analytic Hierarchy Process) de Saaty

#### Principe
L'AHP est une méthode d'aide à la décision basée sur le **jugement d'experts**. Elle demande : *"Entre deux facteurs, lequel contribue le plus aux inondations, et de combien ?"*

#### Comment ça marche ?
On construit une matrice de comparaison par paires (6×6). Par exemple :
- *La pluie est 1,5 fois plus importante que l'altitude pour le risque d'inondation*
- *La pluie est 2 fois plus importante que la pente*

La méthode calcule ensuite les poids de chaque facteur par normalisation mathématique.

#### Vérification de cohérence (CR)
La méthode vérifie que les jugements des experts sont cohérents entre eux. Un **Ratio de Cohérence (CR) < 0,10** est requis.

**Résultat :** CR = **0,0019** → Excellente cohérence (40 fois meilleur que le seuil)

#### Poids calculés par l'AHP expert

| Facteur | Poids | Interprétation |
|---|---|---|
| Pluviométrie | **30,74 %** | Facteur #1 selon les experts |
| Altitude | **20,10 %** | Facteur #2 |
| Drainage | **19,98 %** | Facteur #3 |
| Pente | **13,30 %** | Facteur #4 |
| Sol | **10,59 %** | Facteur #5 |
| Usage | **5,28 %** | Facteur #6 |
| **Total** | **100 %** | |

#### Correction de direction
Un point crucial : certains facteurs ont une logique **inversée** :
- Une **haute altitude** = **bas risque** (pas de basse plaine inondable)
- Une **forte pente** = **eau qui s'écoule** = **bas risque**
- Un **bon drainage** = **bas risque**

Sans cette correction, l'AHP donnait un AUC de seulement 0,37 (pire que le hasard). Avec la correction : **AUC = 0,916**.

---

### Étape 3 — Modèles de Machine Learning

Trois algorithmes apprennent automatiquement les patterns d'inondation depuis les données, **sans règles fixées à l'avance** :

| Algorithme | Principe simplifié |
|---|---|
| **Random Forest** | Forêt de 100 arbres de décision qui votent ensemble |
| **XGBoost** | Arbres construits en corrigeant successivement les erreurs |
| **LightGBM** | Version optimisée de XGBoost, plus rapide |
| **Ensemble Voting** | Vote majoritaire des 3 modèles précédents |

**Split train/test :**
- 3 500 pixels pour apprendre (70 %)
- 1 500 pixels pour tester (30 %)

#### Performances des modèles ML

| Modèle | AUC-ROC | F1-Score | Précision | Rappel |
|---|---|---|---|---|
| Random Forest | **0,950** | 0,889 | 0,925 | 0,856 |
| XGBoost | 0,946 | **0,890** | 0,904 | 0,877 |
| LightGBM | 0,950 | 0,885 | 0,903 | 0,868 |
| Ensemble Voting | 0,950 | 0,885 | 0,903 | 0,868 |

**Poids extraits par le ML (importances moyennes) :**

| Facteur | Poids ML | vs AHP |
|---|---|---|
| Altitude | **26,92 %** | +6,82 % (ML valorise plus l'altitude) |
| Pluviométrie | 18,42 % | −12,32 % (ML la juge moins dominante) |
| Drainage | 17,12 % | −2,86 % |
| Pente | 13,63 % | +0,33 % |
| Sol | 12,99 % | +2,40 % |
| Usage | 10,92 % | +5,64 % |

---

### Étape 4 & 5 — Fusion Hybride AHP + ML

#### Principe de la fusion
On combine les deux sources de connaissance :
- **70 % jugement d'expert** (AHP) → stabilité, cohérence méthodologique
- **30 % données** (ML) → objectivité, correction des biais experts

#### Poids hybrides finaux

| Facteur | AHP Expert | ML Data | **Hybride (final)** |
|---|---|---|---|
| Pluviométrie | 30,74 % | 18,42 % | **27,05 %** |
| Altitude | 20,10 % | 26,92 % | **22,15 %** |
| Drainage | 19,98 % | 17,12 % | **19,12 %** |
| Pente | 13,30 % | 13,63 % | **13,40 %** |
| Sol | 10,59 % | 12,99 % | **11,31 %** |
| Usage | 5,28 % | 10,92 % | **6,97 %** |

---

### Étape 6 — LSTM temporel (Deep Learning)

Le **LSTM (Long Short-Term Memory)** est un réseau de neurones récurrents capable d'apprendre des **séquences temporelles** — ici, l'évolution mois par mois des facteurs hydrologiques.

#### Architecture utilisée
```
Entrée (6 mois d'historique)
    → LSTM(64 neurones, séquentiel)
    → Dropout(20 %)
    → LSTM(32 neurones)
    → Dropout(20 %)
    → Dense(16 neurones)
    → Dense(1 neurone, sigmoid)
    → Probabilité d'inondation le mois suivant
```

**Capacité :** prédit si une zone sera inondée le mois prochain, en apprenant des cycles saisonniers (saison sèche / saison des pluies).

---

### Étape 7 — Carte interactive Folium

La carte HTML générée (`output/carte_pikine_hybrid.html`) présente :

- **Grille choroplèthe** : 35 × 65 cellules colorées selon le niveau de risque
- **Masque polygone** : seule la zone Pikine/Keur Massar est colorée (le reste est transparent)
- **3 couches** sélectionnables :
  - Susceptibilité Hybride (AHP + ML)
  - Probabilité ML pure
  - Zones inondées observées
- **Numéros des 19 communes** affichés
- **Contour du département** visible
- **Légende gradient** : Vert (très bas) → Jaune → Orange → Rouge (très haut)

---

## Analyse des résultats

### Performances comparées des 3 approches

| Méthode | AUC-ROC | F1-Score | Précision | Rappel |
|---|---|---|---|---|
| **AHP Classique** | 0,916 | 0,825 | 0,886 | 0,772 |
| **Hybride AHP+ML** | 0,934 | 0,844 | 0,906 | 0,789 |
| **ML Pur (Ensemble)** | **0,987** | **0,915** | **0,983** | **0,856** |

### Interprétation des métriques

**AUC-ROC** *(Area Under the Curve)* : capacité à distinguer les zones inondées des zones non inondées.
- AUC = 0,50 → pire que le hasard
- AUC = 0,70 → acceptable
- AUC = 0,90 → très bon
- AUC = 1,00 → parfait

**F1-Score** : équilibre entre ne pas manquer de zones inondées (rappel) et ne pas faussement alarmer des zones saines (précision).

### Que retenir ?

#### ✅ AHP classique : Très bon pour une méthode experte pure
- AUC **0,916** est excellent pour une approche sans données d'apprentissage
- La correction de direction (altitude basse = risque haut) était **critique** — sans elle, AUC = 0,37
- Cohérence parfaite (CR = 0,0019 << seuil 0,10)

#### ✅ Hybride : Meilleur compromis
- Amélioration de **+2,0 % d'AUC** par rapport à l'AHP seul
- Intègre le signal des données tout en conservant la logique d'expert
- Recommandé pour la **production** : interprétable + performant

#### ✅ ML pur : Performances maximales
- AUC = **0,987** (quasi-parfait sur données synthétiques)
- Attention : ces performances s'expliquent aussi par le fait que les données synthétiques ont été **générées** avec les mêmes facteurs — sur données réelles les performances seront différentes
- Manque d'interprétabilité pour des décideurs non-techniques

### Analyse des divergences AHP vs ML

La différence la plus importante est sur la **pluviométrie** et l'**altitude** :

- **Expert AHP** : *"La pluie est le facteur #1"* (30,74 %)
  - Logique : pas de pluie = pas d'inondation
- **Machine Learning** : *"L'altitude est le facteur #1"* (26,92 %)
  - Logique : même avec beaucoup de pluie, un terrain élevé ne s'inonde pas

**Ces deux visions sont complémentaires et correctes.** La fusion hybride les concilie : la pluie reste prioritaire (27,05 %) mais l'altitude est réévaluée à la hausse (22,15 %).

### Taux d'inondation par type de sol

| Type de sol | Taux inondé | Explication |
|---|---|---|
| Dunes Littorales | **0 %** | Sol sableux, bien drainé, altitude élevée |
| Ferrugineux tropicaux | 81 % | Zone Est (Keur Massar) + forte pluviométrie |
| Halomorphes | 67 % | Sol salé, drainage limité |
| Hydromorphes | 59 % | Sol gorgé d'eau par nature |
| Eau (lacs/mares) | 31 % | Plans d'eau permanents (déjà comptés) |

> Les Dunes à 0 % d'inondation confirment que la **ceinture dunaire littorale** (communes 15, 16, 17) joue un rôle protecteur naturel. La destruction de ces dunes augmenterait drastiquement le risque pour les zones situées derrière.

---

## Limites et prochaines étapes

### Limites actuelles

1. **Données synthétiques** : Le dataset de 5 000 pixels a été généré informatiquement en s'inspirant des cartes réelles, mais ne remplace pas des mesures terrain. Les performances ML seront inférieures sur données réelles.

2. **Pas de validation terrain** : Il faudrait comparer les prédictions avec des archives d'inondations historiques (photos satellites, rapports de la mairie de Pikine, données ONAS).

3. **Résolution spatiale** : Les cellules de la carte font ~200–300 m de côté. Pour une action précise (rue par rue), une résolution de 10–30 m est nécessaire.

### Prochaines étapes recommandées

| Phase | Action | Données nécessaires |
|---|---|---|
| **Phase 1** | Télécharger les rasters SRTM (altitude réelle) et Sentinel-2 (occupation sol) | Google Earth Engine (gratuit) |
| **Phase 2** | Compiler l'historique des inondations 2015–2024 | Mairie de Pikine, ONAS, CSE Sénégal |
| **Phase 3** | Réentraîner tous les modèles sur données réelles | Ci-dessus |
| **Phase 4** | Déployer une API + dashboard Streamlit + alertes SMS | Python FastAPI + Twilio |

---

## Structure du code

```
elv_Badiane/
├── config.py                  # Paramètres centraux (polygone, bornes, poids Saaty)
├── requirements.txt           # Dépendances Python
│
├── src/
│   ├── data_generator.py      # Génère données synthétiques (distributions réelles)
│   ├── ahp.py                 # Algorithme AHP - matrice Saaty, CR, poids
│   ├── ml_models.py           # RF + XGBoost + LightGBM + Ensemble
│   ├── hybrid.py              # Fusion 70% AHP + 30% ML
│   ├── lstm_model.py          # LSTM temporel (prédiction mensuelle)
│   ├── visualization.py       # Carte Folium + graphiques PNG + rapport
│   └── inference.py           # Prédiction pour un pixel unique
│
├── notebooks/
│   └── poc_notebook.ipynb     # Pipeline complet exécutable pas à pas
│
├── output/
│   ├── carte_pikine_hybrid.html  # ← Carte interactive principale
│   ├── poids_comparison.png      # Graphique comparaison poids AHP/ML/Hybride
│   ├── roc_curves.png            # Courbes ROC des 3 approches
│   ├── confusion_matrices.png    # Matrices de confusion
│   └── rapport_poc.txt           # Rapport texte complet
│
├── models/                    # Modèles sauvegardés (.pkl et .keras)
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── ensemble_model.pkl
│   └── lstm_model.keras
│
└── donneeetude/               # Cartes cartographiques originales (6 facteurs)
    ├── altitude.jpg
    ├── pente.jpg
    ├── pluviometrie.jpg
    ├── drainage.jpg
    ├── sol.jpg
    └── occupation.jpg
```

---

## Conclusion

Ce POC démontre que la combinaison **AHP + Machine Learning** améliore significativement la cartographie des zones inondables par rapport à l'AHP classique seul :

| Indicateur | Valeur | Interprétation |
|---|---|---|
| AUC Hybride | **0,934** | Très bonne discrimination zones inondées/saines |
| Gain vs AHP | **+2,0 %** | Apport mesurable des données sur l'expert |
| CR (cohérence) | **0,0019** | Matrice Saaty parfaitement cohérente |
| Taux inondation | **57,4 %** | Réaliste pour Pikine (très vulnérable) |

La méthode est **reproductible, extensible et prête à être alimentée par des données réelles** dès leur disponibilité. La carte interactive permet aux décideurs de visualiser les niveaux de risque par commune et d'identifier les zones prioritaires d'intervention.

---

*Projet POC — Département de Pikine & Keur Massar, Dakar, Sénégal*
*Méthode : AHP (Saaty) + Random Forest + XGBoost + LightGBM + LSTM*
*Python 3.9 — Anaconda — Folium — scikit-learn — TensorFlow 2.16*
# Explication du projet : Détection des zones inondables par IA
## Pikine & Keur Massar — Fusion AHP + Machine Learning + LSTM

---

## Contexte et problème

Pikine et Keur Massar sont deux communes de la banlieue de Dakar (Sénégal) régulièrement frappées par des inondations catastrophiques, notamment pendant la saison des pluies (juillet–septembre). Des milliers de familles perdent leurs biens chaque année.

**La question posée :**
> *Peut-on cartographier avec précision les zones les plus vulnérables aux inondations, en combinant la méthode AHP classique des experts et l'intelligence artificielle ?*

---

## Les données de base : 6 facteurs de risque

Le projet repose sur **6 facteurs** issus de cartes géographiques réelles de Pikine (planches cartographiques Sy et al.) :

| Facteur | Ce qu'il mesure | Logique de risque |
|---|---|---|
| **Altitude** (m) | Hauteur du terrain (0 – 21,5 m) | Terrain bas → inondation plus facile |
| **Pente** (%) | Inclinaison du terrain (0 – 21,7 %) | Terrain plat → l'eau stagne |
| **Pluviométrie** (mm) | Quantité de pluie reçue (67 – 95 mm) | Plus de pluie → plus de risque |
| **Densité drainage** (km/km²) | Réseau d'évacuation des eaux (0 – 4,6) | Mauvais drainage → eau bloquée |
| **Type de sol** | Nature du sol (5 types) | Sol imperméable → risque élevé |
| **Occupation du sol** | Usage du terrain (6 types) | Zone urbaine → surface imperméable |

### Distributions spatiales réelles observées sur les cartes

```
ALTITUDE  : 0–3 m (80 % de la zone, très plat), dunes au Nord > 8 m
PENTE     : Quasi-nulle (0–0,85 % dominant), légères pentes sur dunes
PLUIE     : Gradient Ouest → Est : 67 mm (Pikine Ouest) → 94 mm (Keur Massar)
DRAINAGE  : Faible (0–0,44 km/km²) dominant, quelques taches à haute densité
SOL       : Hydromorphes dominant (50 %), Ferrugineux (Est), Dunes (Nord)
USAGE     : Culture maraîchère dominant (60 %), Localités (Nord urbain)
```

---

## Le pipeline en 8 étapes

```
  Données       AHP          ML           Fusion       LSTM        Carte
  spatiales  → Expert  →  (RF/XGB/LGB) → Hybride  → Temporel  → Interactive
    [1]         [2]          [3]           [4-5]       [6]          [7]
```

---

### Étape 1 — Génération des données synthétiques

Puisqu'on ne dispose pas encore de données terrain réelles numériques, on génère **5 000 pixels synthétiques** simulant les conditions géographiques réelles de Pikine.

Chaque pixel représente un carré du terrain avec ses 6 valeurs de facteurs, une coordonnée GPS et un label indiquant s'il est inondable ou non.

**Caractéristiques du dataset :**
- 5 000 pixels × 17 colonnes
- Taux d'inondation : **57,4 %** *(Pikine est très vulnérable, ce taux est réaliste)*
- Distributions calées sur les vraies cartes : gradient pluviométrique, zones dunaires, types de sol par zone

---

### Étape 2 — Méthode AHP (Analytic Hierarchy Process) de Saaty

#### Principe
L'AHP est une méthode d'aide à la décision basée sur le **jugement d'experts**. Elle demande : *"Entre deux facteurs, lequel contribue le plus aux inondations, et de combien ?"*

#### Comment ça marche ?
On construit une matrice de comparaison par paires (6×6). Par exemple :
- *La pluie est 1,5 fois plus importante que l'altitude pour le risque d'inondation*
- *La pluie est 2 fois plus importante que la pente*

La méthode calcule ensuite les poids de chaque facteur par normalisation mathématique.

#### Vérification de cohérence (CR)
La méthode vérifie que les jugements des experts sont cohérents entre eux. Un **Ratio de Cohérence (CR) < 0,10** est requis.

**Résultat :** CR = **0,0019** → Excellente cohérence (40 fois meilleur que le seuil)

#### Poids calculés par l'AHP expert

| Facteur | Poids | Interprétation |
|---|---|---|
| Pluviométrie | **30,74 %** | Facteur #1 selon les experts |
| Altitude | **20,10 %** | Facteur #2 |
| Drainage | **19,98 %** | Facteur #3 |
| Pente | **13,30 %** | Facteur #4 |
| Sol | **10,59 %** | Facteur #5 |
| Usage | **5,28 %** | Facteur #6 |
| **Total** | **100 %** | |

#### Correction de direction
Un point crucial : certains facteurs ont une logique **inversée** :
- Une **haute altitude** = **bas risque** (pas de basse plaine inondable)
- Une **forte pente** = **eau qui s'écoule** = **bas risque**
- Un **bon drainage** = **bas risque**

Sans cette correction, l'AHP donnait un AUC de seulement 0,37 (pire que le hasard). Avec la correction : **AUC = 0,916**.

---

### Étape 3 — Modèles de Machine Learning

Trois algorithmes apprennent automatiquement les patterns d'inondation depuis les données, **sans règles fixées à l'avance** :

| Algorithme | Principe simplifié |
|---|---|
| **Random Forest** | Forêt de 100 arbres de décision qui votent ensemble |
| **XGBoost** | Arbres construits en corrigeant successivement les erreurs |
| **LightGBM** | Version optimisée de XGBoost, plus rapide |
| **Ensemble Voting** | Vote majoritaire des 3 modèles précédents |

**Split train/test :**
- 3 500 pixels pour apprendre (70 %)
- 1 500 pixels pour tester (30 %)

#### Performances des modèles ML

| Modèle | AUC-ROC | F1-Score | Précision | Rappel |
|---|---|---|---|---|
| Random Forest | **0,950** | 0,889 | 0,925 | 0,856 |
| XGBoost | 0,946 | **0,890** | 0,904 | 0,877 |
| LightGBM | 0,950 | 0,885 | 0,903 | 0,868 |
| Ensemble Voting | 0,950 | 0,885 | 0,903 | 0,868 |

**Poids extraits par le ML (importances moyennes) :**

| Facteur | Poids ML | vs AHP |
|---|---|---|
| Altitude | **26,92 %** | +6,82 % (ML valorise plus l'altitude) |
| Pluviométrie | 18,42 % | −12,32 % (ML la juge moins dominante) |
| Drainage | 17,12 % | −2,86 % |
| Pente | 13,63 % | +0,33 % |
| Sol | 12,99 % | +2,40 % |
| Usage | 10,92 % | +5,64 % |

---

### Étape 4 & 5 — Fusion Hybride AHP + ML

#### Principe de la fusion
On combine les deux sources de connaissance :
- **70 % jugement d'expert** (AHP) → stabilité, cohérence méthodologique
- **30 % données** (ML) → objectivité, correction des biais experts

#### Poids hybrides finaux

| Facteur | AHP Expert | ML Data | **Hybride (final)** |
|---|---|---|---|
| Pluviométrie | 30,74 % | 18,42 % | **27,05 %** |
| Altitude | 20,10 % | 26,92 % | **22,15 %** |
| Drainage | 19,98 % | 17,12 % | **19,12 %** |
| Pente | 13,30 % | 13,63 % | **13,40 %** |
| Sol | 10,59 % | 12,99 % | **11,31 %** |
| Usage | 5,28 % | 10,92 % | **6,97 %** |

---

### Étape 6 — LSTM temporel (Deep Learning)

Le **LSTM (Long Short-Term Memory)** est un réseau de neurones récurrents capable d'apprendre des **séquences temporelles** — ici, l'évolution mois par mois des facteurs hydrologiques.

#### Architecture utilisée
```
Entrée (6 mois d'historique)
    → LSTM(64 neurones, séquentiel)
    → Dropout(20 %)
    → LSTM(32 neurones)
    → Dropout(20 %)
    → Dense(16 neurones)
    → Dense(1 neurone, sigmoid)
    → Probabilité d'inondation le mois suivant
```

**Capacité :** prédit si une zone sera inondée le mois prochain, en apprenant des cycles saisonniers (saison sèche / saison des pluies).

---

### Étape 7 — Carte interactive Folium

La carte HTML générée (`output/carte_pikine_hybrid.html`) présente :

- **Grille choroplèthe** : 35 × 65 cellules colorées selon le niveau de risque
- **Masque polygone** : seule la zone Pikine/Keur Massar est colorée (le reste est transparent)
- **3 couches** sélectionnables :
  - Susceptibilité Hybride (AHP + ML)
  - Probabilité ML pure
  - Zones inondées observées
- **Numéros des 19 communes** affichés
- **Contour du département** visible
- **Légende gradient** : Vert (très bas) → Jaune → Orange → Rouge (très haut)

---

## Analyse des résultats

### Performances comparées des 3 approches

| Méthode | AUC-ROC | F1-Score | Précision | Rappel |
|---|---|---|---|---|
| **AHP Classique** | 0,916 | 0,825 | 0,886 | 0,772 |
| **Hybride AHP+ML** | 0,934 | 0,844 | 0,906 | 0,789 |
| **ML Pur (Ensemble)** | **0,987** | **0,915** | **0,983** | **0,856** |

### Interprétation des métriques

**AUC-ROC** *(Area Under the Curve)* : capacité à distinguer les zones inondées des zones non inondées.
- AUC = 0,50 → pire que le hasard
- AUC = 0,70 → acceptable
- AUC = 0,90 → très bon
- AUC = 1,00 → parfait

**F1-Score** : équilibre entre ne pas manquer de zones inondées (rappel) et ne pas faussement alarmer des zones saines (précision).

### Que retenir ?

#### ✅ AHP classique : Très bon pour une méthode experte pure
- AUC **0,916** est excellent pour une approche sans données d'apprentissage
- La correction de direction (altitude basse = risque haut) était **critique** — sans elle, AUC = 0,37
- Cohérence parfaite (CR = 0,0019 << seuil 0,10)

#### ✅ Hybride : Meilleur compromis
- Amélioration de **+2,0 % d'AUC** par rapport à l'AHP seul
- Intègre le signal des données tout en conservant la logique d'expert
- Recommandé pour la **production** : interprétable + performant

#### ✅ ML pur : Performances maximales
- AUC = **0,987** (quasi-parfait sur données synthétiques)
- Attention : ces performances s'expliquent aussi par le fait que les données synthétiques ont été **générées** avec les mêmes facteurs — sur données réelles les performances seront différentes
- Manque d'interprétabilité pour des décideurs non-techniques

### Analyse des divergences AHP vs ML

La différence la plus importante est sur la **pluviométrie** et l'**altitude** :

- **Expert AHP** : *"La pluie est le facteur #1"* (30,74 %)
  - Logique : pas de pluie = pas d'inondation
- **Machine Learning** : *"L'altitude est le facteur #1"* (26,92 %)
  - Logique : même avec beaucoup de pluie, un terrain élevé ne s'inonde pas

**Ces deux visions sont complémentaires et correctes.** La fusion hybride les concilie : la pluie reste prioritaire (27,05 %) mais l'altitude est réévaluée à la hausse (22,15 %).

### Taux d'inondation par type de sol

| Type de sol | Taux inondé | Explication |
|---|---|---|
| Dunes Littorales | **0 %** | Sol sableux, bien drainé, altitude élevée |
| Ferrugineux tropicaux | 81 % | Zone Est (Keur Massar) + forte pluviométrie |
| Halomorphes | 67 % | Sol salé, drainage limité |
| Hydromorphes | 59 % | Sol gorgé d'eau par nature |
| Eau (lacs/mares) | 31 % | Plans d'eau permanents (déjà comptés) |

> Les Dunes à 0 % d'inondation confirment que la **ceinture dunaire littorale** (communes 15, 16, 17) joue un rôle protecteur naturel. La destruction de ces dunes augmenterait drastiquement le risque pour les zones situées derrière.

---

## Limites et prochaines étapes

### Limites actuelles

1. **Données synthétiques** : Le dataset de 5 000 pixels a été généré informatiquement en s'inspirant des cartes réelles, mais ne remplace pas des mesures terrain. Les performances ML seront inférieures sur données réelles.

2. **Pas de validation terrain** : Il faudrait comparer les prédictions avec des archives d'inondations historiques (photos satellites, rapports de la mairie de Pikine, données ONAS).

3. **Résolution spatiale** : Les cellules de la carte font ~200–300 m de côté. Pour une action précise (rue par rue), une résolution de 10–30 m est nécessaire.

### Prochaines étapes recommandées

| Phase | Action | Données nécessaires |
|---|---|---|
| **Phase 1** | Télécharger les rasters SRTM (altitude réelle) et Sentinel-2 (occupation sol) | Google Earth Engine (gratuit) |
| **Phase 2** | Compiler l'historique des inondations 2015–2024 | Mairie de Pikine, ONAS, CSE Sénégal |
| **Phase 3** | Réentraîner tous les modèles sur données réelles | Ci-dessus |
| **Phase 4** | Déployer une API + dashboard Streamlit + alertes SMS | Python FastAPI + Twilio |

---

## Structure du code

```
elv_Badiane/
├── config.py                  # Paramètres centraux (polygone, bornes, poids Saaty)
├── requirements.txt           # Dépendances Python
│
├── src/
│   ├── data_generator.py      # Génère données synthétiques (distributions réelles)
│   ├── ahp.py                 # Algorithme AHP - matrice Saaty, CR, poids
│   ├── ml_models.py           # RF + XGBoost + LightGBM + Ensemble
│   ├── hybrid.py              # Fusion 70% AHP + 30% ML
│   ├── lstm_model.py          # LSTM temporel (prédiction mensuelle)
│   ├── visualization.py       # Carte Folium + graphiques PNG + rapport
│   └── inference.py           # Prédiction pour un pixel unique
│
├── notebooks/
│   └── poc_notebook.ipynb     # Pipeline complet exécutable pas à pas
│
├── output/
│   ├── carte_pikine_hybrid.html  # ← Carte interactive principale
│   ├── poids_comparison.png      # Graphique comparaison poids AHP/ML/Hybride
│   ├── roc_curves.png            # Courbes ROC des 3 approches
│   ├── confusion_matrices.png    # Matrices de confusion
│   └── rapport_poc.txt           # Rapport texte complet
│
├── models/                    # Modèles sauvegardés (.pkl et .keras)
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── ensemble_model.pkl
│   └── lstm_model.keras
│
└── donneeetude/               # Cartes cartographiques originales (6 facteurs)
    ├── altitude.jpg
    ├── pente.jpg
    ├── pluviometrie.jpg
    ├── drainage.jpg
    ├── sol.jpg
    └── occupation.jpg
```

---

## Conclusion

Ce POC démontre que la combinaison **AHP + Machine Learning** améliore significativement la cartographie des zones inondables par rapport à l'AHP classique seul :

| Indicateur | Valeur | Interprétation |
|---|---|---|
| AUC Hybride | **0,934** | Très bonne discrimination zones inondées/saines |
| Gain vs AHP | **+2,0 %** | Apport mesurable des données sur l'expert |
| CR (cohérence) | **0,0019** | Matrice Saaty parfaitement cohérente |
| Taux inondation | **57,4 %** | Réaliste pour Pikine (très vulnérable) |

La méthode est **reproductible, extensible et prête à être alimentée par des données réelles** dès leur disponibilité. La carte interactive permet aux décideurs de visualiser les niveaux de risque par commune et d'identifier les zones prioritaires d'intervention.

---

*Projet POC — Département de Pikine & Keur Massar, Dakar, Sénégal*
*Méthode : AHP (Saaty) + Random Forest + XGBoost + LightGBM + LSTM*
*Python 3.9 — Anaconda — Folium — scikit-learn — TensorFlow 2.16*



# 1. Créer et activer l'environnement
python -m venv venv
source venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer le pipeline complet
python main.py

# Ou sans LSTM (plus rapide, ~2 min)
python main.py --skip-lstm

# Ou dans Jupyter
jupyter notebook notebooks/poc_notebook.ipynb



# Installation corrigée (3 options)
# Option 1 — Conda (recommandée, Anaconda déjà installé)

cd /Users/mac/Documents/UCAD_Projet/elv_Badiane

# Créer l'environnement depuis le fichier YAML
conda env create -f environment.yml

# Activer
conda activate ahp_ia_flood

# Lancer le pipeline
python main.py --skip-lstm    # rapide (~2 min)
python main.py                # complet avec LSTM (~10 min)




### Option 2 — Pip direct dans Anaconda (plus simple)

pip install numpy pandas matplotlib seaborn scikit-learn \
            xgboost lightgbm folium tensorflow \
            jupyter notebook ipykernel

# Packages géospatiaux séparément (optionnels, pour données réelles)
conda install -c conda-forge rasterio geopandas shapely


# Option 3 — Utiliser Anaconda base directement

# Depuis la base conda, sans venv
cd /Users/mac/Documents/UCAD_Projet/elv_Badiane
/opt/anaconda3/bin/python main.py --skip-lstmInstallation corrigée (3 options)






