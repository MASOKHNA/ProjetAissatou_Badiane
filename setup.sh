#!/bin/bash
# Script d'installation pour macOS + Anaconda
# Usage : bash setup.sh

set -e
echo "=== Setup POC AHP+IA Pikine & Keur Massar ==="

# Option A : Environnement conda (RECOMMANDE pour macOS)
echo ""
echo "[Option A] Creation environnement conda..."
conda env create -f environment.yml
echo ""
echo "Pour activer : conda activate ahp_ia_flood"
echo "Pour lancer  : python main.py --skip-lstm"
echo ""

# Option B : pip dans un venv existant (si conda pose probleme)
# Decommentez les lignes ci-dessous et commentez Option A
#
# echo "[Option B] Installation pip..."
# pip install numpy pandas matplotlib seaborn scikit-learn
# pip install xgboost lightgbm folium jupyter notebook ipykernel
# pip install tensorflow
# # Geospatial via conda-forge separement :
# conda install -c conda-forge rasterio geopandas shapely
