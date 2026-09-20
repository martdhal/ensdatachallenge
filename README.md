# AML Cohort Explorer

J’ai adapté mon projet ENS de prédiction de survie en une application Streamlit : exploration clinique, prévalence des mutations, estimation de Kaplan–Meier et évaluation d’une baseline Cox.

## Exécution

```bash
git clone https://github.com/martdhal/ensdatachallenge.git
cd ensdatachallenge
```

Interface : **http://localhost:8501**. Commandes exécutées à la racine du dépôt.

### Docker — x86-64

```bash
docker build -t aml-explorer .
docker run --rm -p 127.0.0.1:8501:8501 -v "$(pwd)/data:/app/data:ro" aml-explorer
```

### Docker — Apple Silicon

Exécution AMD64 par émulation sous Docker Desktop ; build ARM64 natif non pris en charge.

```bash
docker buildx build --platform linux/amd64 --load -t aml-explorer .
docker run --rm --platform linux/amd64 -p 127.0.0.1:8501:8501 -v "$(pwd)/data:/app/data:ro" aml-explorer
```

### Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Données et interface

La source se sélectionne dans la barre latérale :

| Source | Données |
| --- | --- |
| **Synthetic demo** (défaut) | 160 patients fictifs ; CSV inclus dans l’image |
| **Repository data** | CSV de `data/` ; volume en lecture seule avec les commandes Docker ci-dessus |
| **Upload CSV files** | CSV clinique requis ; mutations et suivi de survie facultatifs ; schémas illustrés dans `demo/` |

Les filtres portent sur le centre et le pourcentage de blastes. Les onglets donnent accès aux distributions cliniques, aux mutations, aux courbes de survie et au modèle. L’entraînement se déclenche via **Train and evaluate baseline** dans **Model baseline**. Export des données filtrées et des prédictions depuis l’interface.

## Validation et reproductibilité

```bash
# Dans l’environnement Python ci-dessus
python -m pytest -q
```

- Tests : import et validation des CSV, filtrage, modèle et interactions Streamlit.
- [CI GitHub Actions](https://github.com/martdhal/ensdatachallenge/actions) : tests, build Docker et vérification du démarrage du conteneur à chaque push et pull request.
- Dépendances figées dans `requirements.txt` ; graine 42 pour la démonstration et la séparation entraînement/test.
