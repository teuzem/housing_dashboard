## App Tableau de bord Analytique des prix des biens immobiliers EDA.

## Structure du projet

```
housing_eda/
├── Housing.csv              # Dataset source
├── housing_app.py           #  Application Streamlit
├── EDA_Housing_Price_NGOUMTSOP.ipynb # Notebook Jupiter d'analyse EDA, statistique et visualisations
├── requirements.txt         # Dépendances Python
└── /outputs/figures/         # Graphiques générés par EDA_Housing_Price_NGOUMTSOP.ipynb
    ├── price_distribution.png
    ├── univariate_numeric.png
    ├── ...
    └── Distribution du prix des biens immobiliers.png
└── /outputs/tables/         # Tables de resultats generes par EDA_Housing_Price_NGOUMTSOP.ipynb
    ├── stats_descriptives.csv
    ├── frequences_binaires.csv
    ├── ...
    └── synthese_EDA_Housing.csv
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
streamlit run housing_app.py
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
   - Choisir le fichier `housing_app.py` comme Main file
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

### Ouvrir le fichier EDA_Housing_Price_NGOUMTSOP.ipynb et cliquer sur Run all si votre kernel est pret.
