# Classification de tumeurs mammaires à partir de mammographies

## Description
Ce projet vise à détecter et classifier automatiquement des tumeurs mammaires en utilisant des mammographies, afin d'aider les radiologues à :
- Identifier les tumeurs (bénignes ou malignes).
- Réduire les faux négatifs.
- Prioriser les cas nécessitant une attention urgente.

Le projet utilise des techniques d'apprentissage profond, notamment des réseaux de neurones convolutionnels (CNN), pour analyser les images médicales.

---

## Fonctionnalités
- Classification des images en trois catégories :
  - Aucun cancer détecté.
  - Cancer bénin détecté.
  - Cancer malin détecté.
- Prédiction de la probabilité de malignité pour chaque image.
- Préparation pour une future évolution vers la segmentation d'images.

---

## Structure du projet
P1-Classification/
│
├── data/               # Dossier contenant les données (radiographies)
├── notebooks/          # Notebooks Jupyter pour l'exploration et le prototypage
├── scripts/            # Scripts Python pour le traitement des données et l'entraînement du modèle
├── P1/                 # Environnement virtuel pour isoler les dépendances
├── requirements.txt    # Liste des bibliothèques utilisées
└── README.md           # Documentation du projet

## Données
Les radiographies utilisées dans ce projet proviennent de datasets publics :

CBIS-DDSM : Dataset de mammographies avec annotations précises (bénin/malin).

## Objectifs du projet
Étape 1 : Préparation des données

Charger, explorer, et prétraiter les radiographies.
Appliquer des techniques de normalisation et de mise à l’échelle.
Étape 2 : Entraînement du modèle

Implémenter un modèle CNN pour la classification.
Entraîner le modèle sur les données prétraitées.
Étape 3 : Optimisation

Utiliser des architectures avancées (ResNet, EfficientNet).
Améliorer la gestion des données déséquilibrées.
Étape 4 : Interprétabilité

Intégrer des outils d’explicabilité (Grad-CAM, SHAP, LIME).
Visualiser les zones d’intérêt des mammographies.

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

pip install -r requirements.txt
### Lancer le projet
Pour explorer les données ou tester le modèle, ouvre un notebook Jupyter :

jupyter lab
Pour exécuter un script Python, utilise :

python scripts/<nom_du_script>.py
