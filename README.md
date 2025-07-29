# P1-Classification
# Classification de tumeurs mammaires à partir de mammographies

## Description
Ce projet vise à détecter et classifier automatiquement des tumeurs mammaires en utilisant des mammographies, afin d'aider les radiologues à :
- Identifier les tumeurs (bénignes ou malignes).
- Réduire les faux négatifs.
- Prioriser les cas nécessitant une attention urgente.

Le projet utilise des techniques de deep learning, notamment des réseaux de neurones convolutionnels (CNN), pour analyser les images médicales.

---

## Fonctionnalités
- Classification des images en trois catégories :
  - Tumeur bénigne sans retour.
  - Tumeur bénigne.
  - Tumeur maligne (Cancer).
- Prédiction de la probabilité de malignité pour chaque image.
- Préparation pour une future évolution vers la segmentation d'images.

---

## Structure du projet!

P1-Classification/

├── P1/ # Modules complémentaires (prétraitements, métriques, etc.)

├── analysis_results/ # Résultats d’analyse exploratoire, courbes, logs, métriques

├── app/ # Application (ex : FastAPI, Streamlit ou Flask)

├── notebook/ # Notebooks Jupyter pour l’exploration et les tests

├── scripts/ # Scripts de traitement, d'entraînement ou de pipeline

├── .gitignore # Fichiers à exclure du versionnement Git

├── app.py # Fichier principal de lancement de l'application

├── installed_packages.txt # Liste figée des dépendances installées

└── README.md # Documentation du projet

## Données
Les radiographies utilisées dans ce projet proviennent de datasets publics :

CBIS-DDSM : Dataset de mammographies avec annotations précises (bénin/malin).

## Objectifs du projet
### Étape 1 : Préparation des données

Charger, explorer, et prétraiter les radiographies.
Appliquer des techniques de normalisation et de mise à l’échelle.

### Étape 2 : Entraînement du modèle

Implémenter un modèle CNN pour la classification.
Entraîner le modèle sur les données prétraitées.

### Étape 3 : Optimisation

Test de plusieurs modèles (EfficientNetBO, MobileNetV3Small, ...)
Varier les hyper paramètres à prendre en compte
Retravailler les données de bases pour avoir la meilleure interprétabilité possible.
#### Aller à la fin de model.ipynb pour voir les résultats de la formule gardée

### Étape 4 : Application

Création d'une application Web "Mammo Analyzer" qui permet de détecter quel type de tumeur
est présent sur la mammographie.
<img width="1415" alt="image" src="https://github.com/user-attachments/assets/5dbd9e2c-9a98-452a-aaf3-b39ed836696f" />


## Démarrage rapide
### Prérequis
Python 3.10 ou une version compatible.
Un environnement virtuel configuré (optionnel mais recommandé).
Les bibliothèques suivantes doivent être installées :
numpy
pandas
matplotlib
opencv-python
tensorflow
keras
jupyterlab
### Installation
Clone ce dépôt :

git clone https://github.com/ton-repo/P1-Classification.git

cd P1-Classification

Crée et active un environnement virtuel (optionnel) :

python3 -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows

Installe les dépendances :

pip install -r installed_packages.txt

### Lancer le projet
Pour explorer les données ou tester le modèle, ouvre un notebook Jupyter :

jupyter lab
Pour exécuter un script Python, utilise :

python scripts/<nom_du_script>.py
