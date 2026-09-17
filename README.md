# AML Cohort Explorer

J’ai adapté mon projet ENS de prédiction de survie en une application Streamlit pour explorer les données cliniques, les mutations et les courbes de survie. J’y propose également une baseline Cox simplifiée.

## Récupérer le projet

```bash
git clone https://github.com/martdhal/ensdatachallenge.git
cd ensdatachallenge
```

Toutes les commandes suivantes s’exécutent à la racine du dépôt.

## Lancer avec Docker

Docker doit être démarré (Docker Desktop sur Mac). Les commandes ci-dessous montent les données du dépôt en lecture seule.

### Mac Apple Silicon (M1 et suivants)

J’ai utilisé ces commandes sur mon Mac. L’image AMD64 s’exécute par émulation ; la construction ARM64 native n’est pas prise en charge.

```bash
docker buildx build --platform linux/amd64 --load -t aml-explorer .
docker run --rm --platform linux/amd64 -p 127.0.0.1:8501:8501 -v "$(pwd)/data:/app/data:ro" aml-explorer
```

### Intel / AMD (x86-64)

```bash
docker build -t aml-explorer .
docker run --rm -p 127.0.0.1:8501:8501 -v "$(pwd)/data:/app/data:ro" aml-explorer
```

Ouvrir **http://localhost:8501**. Arrêter avec **Ctrl+C**.

## Alternative : lancement avec Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

L’interface est accessible à la même adresse : **http://localhost:8501**.

## Utiliser l’interface

Dans la barre latérale, choisir une source :

- **Synthetic demo** : démonstration par défaut sur 160 patients fictifs.
- **Repository data** : données du projet dans `data/`, accessibles avec les commandes ci-dessus.
- **Upload CSV files** : import de données personnelles au format des exemples de `demo/` ; fichier clinique obligatoire, mutations et suivi de survie facultatifs.

Filtrer les centres et le pourcentage de blastes, puis consulter les onglets **Clinical overview**, **Mutations**, **Survival** et **Model baseline**. Dans ce dernier, cliquer sur **Train and evaluate baseline** pour entraîner le modèle et afficher son score. Les tableaux filtrés et les prédictions peuvent être téléchargés depuis l’interface.

Aucun compte ni clé API n’est nécessaire. L’image Docker contient la démonstration synthétique ; les données du projet sont fournies par le montage du dossier `data/`.

## Tests et CI

Dans l’environnement Python installé ci-dessus :

```bash
python -m pytest -q
```

Les tests couvrent notamment l’import et la validation des CSV, les filtres, le modèle et l’interface. La [CI GitHub Actions](https://github.com/martdhal/ensdatachallenge/actions) exécute les tests, construit l’image Docker et vérifie le démarrage de l’application à chaque push et pull request.

Les dépendances sont figées dans `requirements.txt`. La génération de la démonstration et la séparation entraînement/test utilisent la graine 42.
