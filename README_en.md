<div align="center">

# 🏸 Happy Badminton

**Badminton match win-probability predictor**

[![AUC](https://img.shields.io/badge/AUC-0.9608-brightgreen?style=for-the-badge&logo=python)](README_en.md)
[![Accuracy](https://img.shields.io/badge/Accuracy-89.7%25-blue?style=for-the-badge)](README_en.md)
[![Tests](https://img.shields.io/badge/Tests-150%20passing-success?style=for-the-badge&logo=pytest)](README_en.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](README_en.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](README_en.md)

*GBM Stacking Ensemble — LightGBM + XGBoost + CatBoost + BayesianRidge meta-learner + TemperatureScaler calibration*

[中文 README](README.md) · [Quick Start](#quick-start) · [Performance](#model-performance) · [Acknowledgements](#acknowledgements)

</div>

---

## Quick Start

```bash
uv sync                   # Install dependencies
uv run python main.py     # Auto-train on first run (~3–5 min), then start server
# Open http://localhost:5001
```

<details>
<summary>More commands</summary>

```bash
uv run python main.py --train      # Force re-train
uv run python main.py --port 8080  # Custom port
./run.sh                           # Shell shortcut (Mac/Linux)

# Train / validate standalone
uv run python scripts/train_simplified.py
uv run python scripts/validate_model.py
uv run python scripts/optimize_sota.py      # Optuna hyperparameter search (optional)

# Quality gate
uv run ruff check . && uv run ruff format . && uv run pytest tests/ -v
```

</details>

---

## Model Performance

<div align="center">

| Metric | Value |
|:------:|:-----:|
| 🎯 **ROC AUC** | **0.9608** |
| 📉 LogLoss | 0.2316 |
| 📐 Brier Score | 0.0722 |
| ✅ Accuracy | 89.7% |
| 🏆 Winner Recall | 93.0% |
| 🔍 Upset Detection | 82.3% |

*Test set: 5,704 matches · Strict time-based split 70/15/15*

</div>

### Evaluation Charts

<table align="center">
<tr>
<td align="center"><img src="docs/plots/eval_roc_curve.png" width="340"/><br><b>ROC Curve</b><br><sub>AUC=0.9608, far above the random baseline</sub></td>
<td align="center"><img src="docs/plots/eval_pr_curve.png" width="340"/><br><b>Precision-Recall Curve</b><br><sub>High precision and recall simultaneously</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/plots/eval_confusion_matrix.png" width="340"/><br><b>Confusion Matrix</b><br><sub>Winner recall 93.0%, upset detection 82.3%</sub></td>
<td align="center"><img src="docs/plots/eval_calibration.png" width="340"/><br><b>Calibration Curve</b><br><sub>Predicted probabilities closely track actual win rates</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/plots/eval_score_distribution.png" width="340"/><br><b>Score Distribution</b><br><sub>Two clearly separated peaks for winners and losers</sub></td>
<td align="center"><img src="docs/plots/eval_feature_importance.png" width="340"/><br><b>Feature Importance Top 20</b><br><sub>ELO difference and recent form are the strongest signals</sub></td>
</tr>
</table>

---

## Architecture

```mermaid
flowchart TD
    A[📊 Raw Data\nTournament Results.xlsx] --> B[preprocess_pipeline\nFilter retirements · parse scores]
    B --> C[build_advanced_features\nMOV Elo · Form · H2H · Fatigue · Nationality]
    C --> D[FeatureEngineer\nStandardisation · Rolling shift-1]
    D --> E{Time-based split 70/15/15}
    E -->|Train 70%| F[Base Models\nLightGBM · XGBoost · CatBoost]
    E -->|Val 15%| F
    F -->|Meta features| G[Meta Model\nBayesianRidge]
    G --> H[TemperatureScaler\nProbability calibration]
    H --> I[🎯 predict_proba_calibrated\nAUC 0.9608]
    E -->|Test 15%| I
```

---

## Frontend Predictor

Visit `http://localhost:5001` and enter pre-match stats to get a prediction.

| Mode | Inputs | Best for |
|------|--------|----------|
| ⚡ **Quick** | Ranking, ELO, nationality, level, round, home ground | Fast judgement in 30 seconds |
| 🔬 **Expert** | All fields required: ranking, ELO, nationality, level, round, host country, form (5/10/20), H2H, streak, career matches, 3-set rate | Maximum accuracy |

Stats lookup: [BadmintonRanks.com](https://badmintonranks.com)

---

## 35 Pre-match Features

<details>
<summary>Expand full feature list</summary>

| Category | Features |
|----------|----------|
| 📊 Ranking | `log_rank_diff`, `rank_closeness` |
| 🏟️ Match context | `category_flag`, `level_numeric`, `round_stage`, `match_month` |
| 🏠 Home advantage | `winner_home`, `loser_home`, `level_x_home`, `home_x_closeness` |
| 📈 Recent form | `winner_form_5`, `loser_form_5`, `form_diff_5/10/20` |
| ⚡ Momentum | `form_momentum_w`, `form_momentum_l`, `momentum_diff` |
| 🔥 Streak (capped ±5) | `streak_capped_w`, `streak_capped_l`, `streak_capped_diff` |
| ⚔️ H2H | `h2h_win_rate_bayes` (Bayesian-smoothed) |
| 🎖️ Experience | `career_stage`, `career_stage_l` (U-curve) |
| 🔗 Interactions | `rank_x_form_diff`, `rank_closeness_x_h2h`, `gender_x_rank` |
| 🌏 Nationality | `same_nationality`, `nat_matchup_win_diff` |
| 🌐 Continental home | `winner_continent_home`, `loser_continent_home`, `continent_advantage_diff` |
| 📡 ELO | `winner_elo`, `loser_elo`, `elo_diff` |

</details>

---

## File Structure

<details>
<summary>Expand</summary>

```
Happy-Badminton/
├── main.py                        # One-click entry (auto-train + start)
├── config.yaml                    # All hyperparameters (single source of truth)
├── data/raw/
│   ├── Tournament Results.xlsx    # Full data (gitignored)
│   └── Tournament Results - Sample.xlsx  # 20-match sample
├── src/
│   ├── data/
│   │   ├── loader.py              # Excel loading & merging
│   │   ├── preprocessor.py        # Cleaning & outlier handling
│   │   ├── advanced_features.py   # MOVEloRating · MomentumFeatures · H2H · Nationality
│   │   └── feature_engineering.py # Standardisation · Rolling (all shift(1))
│   ├── models/
│   │   └── ensemble_models.py     # StackingEnsemble + TemperatureScaler
│   └── utils/
│       └── constants.py           # LEAK_FEATURES · LEVEL_MAP (single source of truth)
├── scripts/
│   ├── train_simplified.py        # Train general predictor
│   ├── validate_model.py          # Test-set evaluation
│   └── generate_eval_plots.py     # Generate evaluation charts
├── frontend/
│   ├── app.py                     # Flask API
│   ├── templates/index.html       # 4-view SPA
│   └── static/                    # JS · CSS · SVG
├── models/                        # Trained artefacts (gitignored)
└── tests/                         # 180+ pytest tests
```

</details>

---

## Dataset

| Field | Value |
|-------|-------|
| 📅 Date range | 2022-01-11 to 2025-01-16 |
| 🎾 Matches | 38,024 (after filtering retirements) |
| 👤 Players | 5,293 |
| 🏆 Levels | OG · WC · WTF · S1000 · S750 · S500 · S300 · S100 · IS · IC |

---

## Acknowledgements

<div align="center">

**All training data provided by [BadmintonRanks.com](https://badmintonranks.com/)**

A huge thank you to the owner of BadmintonRanks.com for generously granting permission
to use their database for this research project.
This project simply would not exist without their support.
If you're working with badminton data, [BadmintonRanks.com](https://badmintonranks.com) is the place to start. 🙏

</div>

---

<div align="center">

MIT License · Made with ❤️ and a lot of shuttlecocks

</div>
