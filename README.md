# App Tableau de bord Analytique des prix des biens immobiliers EDA.

## Structure du projet

```
housing_eda/
├── Housing.csv              # Dataset source
├── housing_eda.py           # Script d'analyse EDA complet (génère les figures)
├── app.py                   # Application Streamlit
├── requirements.txt         # Dépendances Python
└── figures/                 # Graphiques générés par housing_eda.py
    ├── 01_price_distribution.png
    ├── 02_univariate_numeric.png
    ├── ...
    └── stats_descriptives.csv
```
## Prerequis avant lancement
## Installer et activer un environnement virtuel python pour Streamlit
```bash
# Creer l'environnement virtuel
python -m venv nom_venv

#Activer l'environnement
source nom_venv/Scripts/activate
```
## Lancement en local

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'application
streamlit run app.py
```
## Déploiement sur Streamlit Cloud (gratuit)

1. Créer un compte sur https://streamlit.io/cloud
2. Créer un nouveau dépôt GitHub et y pousser les fichiers :
   - `housing_app.py`
   - `Housing.csv`
   - `requirements.txt`

3. Dans Streamlit Cloud :
   - Cliquer "New app"
   - Sélectionner votre dépôt GitHub
   - Choisir le fichier `app.py` comme Main file
   - Cliquer "Deploy"

L'application sera accessible via une URL publique.

## Contenu de l'analyse EDA

- **Vue d'ensemble** : KPIs, distribution du prix, segments
- **Analyse univariée** : distribution individuelle de chaque variable
- **Analyse bivariée** : corrélations, prix par catégorie, régressions
- **Analyse multivariée** : interactions, heatmaps croisées, profils de segments
- **Insights décisionnels** : impact des équipements, recommandations, tableaux de synthèse
- **Export des donnees**: Exportez rapidement toutes les tables d'analyse 

## Execution du notebook Jupiter de l'EDA et Visualisation

# Ouvrir le fichier EDA_Housing_Price_NGOUMTSOP.ipynb et cliquer sur Run all si votre kernel est pret.