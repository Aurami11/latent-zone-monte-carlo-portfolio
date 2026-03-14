# Energy-Based Top-K Multi-Asset Portfolio

```md
Research prototype — latent-zone price modeling, Monte Carlo barrier probabilities, Top-K multi-asset allocation
````

Prototype de recherche en finance quantitative explorant une modélisation des prix par **zones latentes**, **tubes quantiles**, **enveloppes de volatilité** et **probabilités de franchissement estimées par simulation Monte Carlo**.

## Overview

Ce projet propose une modélisation des prix financiers fondée sur une analogie énergétique :

- le prix observé évolue dans un ensemble de **zones de prix** construites à partir de quantiles glissants ;
- un **état latent** représente la zone effectivement retenue par le modèle ;
- une **énergie instantanée** mesure l’intensité du mouvement observé relativement à la volatilité locale ;
- une probabilité de transition entre zones est calculée afin de filtrer certains débordements jugés non validés ;
- les probabilités de franchissement haussier/baissier sont ensuite estimées par **Monte Carlo** ;
- enfin, une stratégie **Top-K long-only** sélectionne les actifs présentant le meilleur compromis entre probabilité de percée haussière et potentiel haussier résiduel.

Ce dépôt correspond à un **proof of concept** et non à un système de trading en production.

## Main ideas

### 1. Log-price modeling

Le modèle travaille sur :

\[
x_t = \log(P_t)
\]

afin de rendre les variations additives et de faciliter la comparaison entre actifs.

### 2. Quantile tubes

Plusieurs tubes de prix sont construits à partir de quantiles glissants du log-prix :

- tube central ;
- tubes intermédiaires ;
- tube externe.

Ils servent à représenter des zones de marché plus ou moins extrêmes.

### 3. Volatility shells

Des extensions basées sur la volatilité locale sont ajoutées autour du tube externe afin d’autoriser des sorties au-delà de l’historique empirique récent.

### 4. Observed zone vs latent zone

Le prix observé peut se situer dans une zone différente de celle retenue par le modèle.

L’état latent est mis à jour de manière probabiliste à partir :

- de l’énergie instantanée ;
- d’un seuil énergétique local ;
- de la distance entre zone observée et zone latente.

### 5. Coherent projection into the latent zone

Quand la zone observée diffère de la zone latente, le modèle ne démarre pas la simulation depuis le prix observé brut.

À la place, il **projette la position observée dans la zone latente en conservant sa position relative**.  
Cela permet de :

- préserver l’information directionnelle ;
- éviter les biais liés à un simple clipping sur les bornes ;
- garder une cohérence entre filtre latent et Monte Carlo.

### 6. Monte Carlo barrier probabilities

À partir du point reconstruit dans la zone latente, le modèle estime par simulation Monte Carlo :

- la probabilité d’atteindre la borne supérieure avant la borne inférieure ;
- la probabilité d’atteindre la borne inférieure avant la borne supérieure ;
- la probabilité de rester à l’intérieur de la zone sur l’horizon considéré.

### 7. Top-K portfolio construction

La stratégie est **long-only**.

À chaque date de réallocation :

- les actifs sont scorés selon leur probabilité de percée haussière ;
- on ajoute un terme de **potentiel haussier résiduel** depuis le point latent reconstruit ;
- les **Top-K** actifs sont retenus ;
- les poids sont alloués proportionnellement au score.

## Repository structure

```text
.
├── config.py
├── core.py
├── yfinance_data.py
├── utility_func.py
├── plot_func.py
├── portfolio_topk_horizon.py
├── report_model.tex
├── README.md
├── yf_cache/
└── bt_topk_horizon_outputs/
    └── topk_horizon_portfolio/
````

## Files description

### `config.py`

Contient les paramètres globaux du projet :

* univers d’actifs ;
* dates ;
* paramètres des tubes quantiles ;
* paramètres de volatilité ;
* paramètres du filtre latent ;
* paramètres Monte Carlo ;
* paramètres portefeuille et backtest.

### `core.py`

Contient les fonctions principales du modèle :

* construction des tubes quantiles ;
* calcul de l’état du marché ;
* construction des coquilles de volatilité ;
* classification des zones ;
* filtre latent ;
* estimation de la dynamique locale ;
* simulation Monte Carlo ;
* construction des signaux ;
* construction des pondérations Top-K ;
* backtest.

### `yfinance_data.py`

Gère la récupération des données Yahoo Finance :

* vérifie si les fichiers `.parquet` sont déjà présents dans le cache ;
* sinon télécharge les données ;
* renvoie les données OHLC nécessaires au projet.

### `utility_func.py`

Regroupe les fonctions utilitaires :

* fonctions de nommage ;
* gestion de dates ;
* statistiques annuelles ;
* fonctions d’aide diverses utilisées dans le pipeline.

### `plot_func.py`

Contient les fonctions de visualisation :

* graphique de comparaison de courbes d’equity ;
* graphique des poids du portefeuille ;
* autres fonctions de tracé utilisées pour les sorties.

### `portfolio_topk_horizon.py`

Script principal du projet :

* charge les données ;
* construit les signaux ;
* génère les poids ;
* lance le backtest ;
* compare la stratégie aux benchmarks ;
* exporte les résultats dans :

```text
bt_topk_horizon_outputs/topk_horizon_portfolio/
```

### `yf_cache/`

Dossier de cache pour les données téléchargées depuis Yahoo Finance.

### `bt_topk_horizon_outputs/`

Dossier de sortie contenant :

* les signaux ;
* les pondérations cibles ;
* les diagnostics de réallocation ;
* les courbes d’equity ;
* les poids journaliers ;
* les benchmarks ;
* les résumés de performance ;
* les figures exportées.

## Installation

Créer un environnement Python puis installer les dépendances :

```bash
pip install yfinance pandas numpy matplotlib pyarrow
```

## Run

Lancer le script principal :

```bash
python portfolio_topk_horizon.py
```

## Outputs

Les résultats sont enregistrés dans :

```text
bt_topk_horizon_outputs/topk_horizon_portfolio/
```

Le dossier contient notamment :

* `all_signals.csv`
* `strategy_target_weights.csv`
* `strategy_rebalance_diagnostics.csv`
* `strategy_equity.csv`
* `strategy_daily_weights.csv`
* `strategy_invested_weight.csv`
* `equal_weight_equity.csv`
* `spy_ief_60_40_equity.csv`
* `spy_buyhold_equity.csv`
* `benchmark_curves.csv`
* `performance_summary.csv`
* `portfolio_vs_benchmarks.png`
* `strategy_daily_weights.png`

## Dependencies

* Python
* pandas
* numpy
* matplotlib
* yfinance
* pyarrow

## Notes on methodology

Le projet repose sur plusieurs choix méthodologiques importants :

* utilisation du **log-prix** plutôt que du prix brut ;
* segmentation du marché en **zones** à partir de tubes quantiles ;
* ajout d’extensions de volatilité pour représenter des sorties extrêmes ;
* distinction entre **zone observée** et **zone latente** ;
* reconstruction cohérente du point de départ dans la zone latente avant simulation ;
* construction d’un score portefeuille fondé sur la probabilité de percée haussière et le potentiel résiduel.

## Limitations

Ce projet est un **prototype de recherche**. Il présente plusieurs limites :

* paramètres encore largement heuristiques ;
* absence de calibration statistique complète ;
* absence de coûts de transaction réalistes dans la version de base ;
* dépendances croisées entre actifs non modélisées explicitement ;
* sensibilité possible aux choix de fenêtres, seuils et quantiles ;
* les performances passées ne préjugent pas des performances futures.

## Warning

Ce dépôt est fourni à des fins de recherche, d’apprentissage et d’expérimentation.
Il ne constitue ni un conseil en investissement, ni un système de trading prêt à l’emploi, ni une promesse de performance.

## Documentation

Le dépôt peut être accompagné d’un rapport LaTeX expliquant :

* la logique des tubes quantiles ;
* la construction de l’état latent ;
* la probabilité de transition entre zones ;
* la projection relative entre zone observée et zone latente ;
* la logique de sélection du portefeuille Top-K.

## Possible improvements

* calibration plus rigoureuse des paramètres ;
* comparaison plus poussée entre drift linéaire et drift non linéaire ;
* introduction de dynamiques à sauts ou à bruit non gaussien ;
* prise en compte explicite des corrélations inter-actifs ;
* validation hors échantillon plus poussée ;
* étude de robustesse sur plusieurs régimes de marché.

## Author

Projet personnel de recherche en modélisation quantitative et allocation de portefeuille.

````