# 🌾 Agri-Desease-Detect-AI

Détection de maladies agricoles par image 📷  
Projet d'intelligence artificielle intégrée dans une application mobile Flutter, destinée aux agriculteurs pour diagnostiquer les plantes localement, **sans connexion internet**.


## 📦 Structure du projet

Architecture modulaire : séparation entre IA (Python) et app mobile (Flutter).
ci dessos est une image monrant comment le project foctionne 

![Vue d'ensemble](https://raw.githubusercontent.com/ton-utilisateur/ton-repo/main/docs/diagramme.png)


---

## 🚀 Installation & Configuration

### 🧠 IA – Partie Python (dossier racine du projet)

Toutes les commandes suivantes doivent être exécutées à la **racine du projet `agri-desease-detect-ai/`**, là où se trouvent les dossiers `scripts/`, `data/`, `model/`, etc.

```bash
# 1. Créer un environnement virtuel Python
python -m venv venv

# 2. Activer l'environnement virtuel
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Installer les dépendances IA
pip install -r requirements.txt

---

###  – Partie mobile (dossier agri_desease_detect_app)

# 1. Installer les dépendances Flutter
flutter pub get

# 3. Lancer l’application sur un émulateur ou un téléphone
flutter run
