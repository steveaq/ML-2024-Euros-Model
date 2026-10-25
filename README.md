# EURO 2024 model notes

A multipage Streamlit app built from my original EURO 2024 notebooks. It follows the analysis from public results and FIFA rankings through feature engineering, separate goal regressions, Poisson match probabilities, tournament simulation and a shareable report.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or later is recommended. The date-frozen datasets are bundled; the app does not need API keys or Kaggle authentication. The first visit fits the selected model, then caches its results.

## Pages

- **Overview:** the modelling question and a score-probability example.
- **Data & coverage:** source records, date exclusions, goal distributions and provenance.
- **Feature engineering:** pre-match form, rank weighting and feature assumptions.
- **Model bench:** mean, linear, Random Forest and gradient boosting comparisons on 2023.
- **Validation:** chronological 2024 holdout errors, residuals, reliability and predictions.
- **Match lab:** neutral matchups, goal distributions, score grids and sensitivity scenarios.
- **Tournament:** seeded group and knockout simulations, stage probabilities and simulation error.
- **Presentation:** results, actual group-game review and a standalone HTML report download.

This is a retrospective reconstruction, not a preserved pre-tournament forecast. The original input CSVs and fitted pickles were never committed. The notebooks are retained; [method notes](docs/METHODS.md) explain what changed. In particular, the Gamma notebook actually compares linear and tree regressors; it does not fit a Gamma or XGBoost model.

## Data and reproducibility

International results come from [Mart Jürisoo's public dataset](https://github.com/martj42/international_results). FIFA rankings use the April 2024 [cashncarry dataset](https://www.kaggle.com/datasets/cashncarry/fifaworldranking), with the exact mirror URL and both raw/bundled SHA-256 hashes in [the manifest](data/manifest.json). Inputs are pinned to repository commits. Dataset licences and source attribution remain with their respective publishers.

```bash
python scripts/prepare_data.py
python -m unittest discover -s tests -v
python scripts/export_report.py --output euro-2024-report.html
```

Preparation downloads the pinned inputs if absent and builds compact local snapshots. The normal app reads bundled snapshots without downloading. Results before 14 June 2024 enter modelling; actual EURO group games are kept for retrospective evaluation. Some historical scores include extra time, which limits their use as a 90-minute target. The app makes that assumption visible.

## Streamlit Community Cloud

Connect this repository, select the `main` branch and set the entrypoint to `app.py`. No secrets are required. Dependencies are in `requirements.txt`; theme settings are in `.streamlit/config.toml`. Data and model calculations run locally in the Streamlit process. Simulations are cached by model settings, run count and seed; shared cache entries contain only public research data.

## Design

An editorial research layout based on the [Editorial skill in awesome-design-skills](https://github.com/bergside/awesome-design-skills/tree/main/skills/editorial), with accessible controls, labelled charts and restrained writing. See [design notes](docs/DESIGN.md).
