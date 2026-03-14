---
title: Happy Badminton
emoji: 🏸
colorFrom: indigo
sdk: docker
pinned: false
license: mit
---

# Happy Badminton - AI Match Prediction

[![Badminton](https://img.shields.io/badge/Badminton-AI%20Prediction-blue)](https://huggingface.co/spaces/owenlee-5678/happy-badminton)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green)](https://flask.palletsprojects.com/)
[![Model](https://img.shields.io/badge/Models-HuggingFace-orange)](https://huggingface.co/owenlee-5678/happy-badminton-models)

AI-powered badminton match prediction using ensemble machine learning models (LightGBM + XGBoost + CatBoost → BayesianRidge meta-learner).

## 🎯 Features

- **Match Winner Prediction**: Predict which player will win with probability confidence
- **Set Count Prediction**: Forecast whether a match will go to 2 or 3 sets
- **Real-time Inference**: Fast predictions powered by pre-trained models
- **Driving Factors Analysis**: See which features most influence each prediction

## 🚀 How to Use

1. Visit the Space homepage
2. Enter match details for both players (ranking, form, streak, etc.)
3. Click "Predict Winner"
4. View results with confidence interval and driving factors

## 📊 Model Performance

### Main Prediction Model
- **AUC**: 0.9608
- **LogLoss**: 0.2316
- **Brier Score**: 0.0722
- **Features**: 35 pre-match features

### Set Count Model
- **AUC**: 0.6635
- **LogLoss**: 0.5583

## 🧠 Model Details

Models are automatically downloaded from [HuggingFace Model Hub](https://huggingface.co/owenlee-5678/happy-badminton-models) on first launch.

**Model Files**:
- `simplified_ensemble.pkl` (3.3MB) - Main prediction model
- `set_count_model.pkl` (1.1MB) - Set count prediction model
- Feature metadata and nationality pair lookup tables

## 📖 Input Data Required

All player stats should be sourced from [BadmintonRanks.com](https://badmintonranks.com):

### Required Fields (Quick Mode)
- Player names, nationality, BWF ranking
- Recent form (last 5/10/20 matches)
- ELO rating (optional, defaults to 1616)

### Additional Fields (Expert Mode)
- H2H record (wins/total)
- Current win/loss streak
- Career total matches
- Historical 3-set match rate
- Host country (for home advantage)

## 🔧 Technical Stack

- **Backend**: Flask 3.0+
- **ML**: scikit-learn, LightGBM, XGBoost, CatBoost
- **Models**: Custom stacking ensemble with BayesianRidge meta-learner
- **Calibration**: Temperature Scaling for probability calibration
- **Frontend**: Vanilla JavaScript (no framework dependencies)

## 📦 Training Data

- **Source**: BWF official tournament records (2019-2025)
- **Matches**: ~15,000 professional matches
- **Time-based split**: 70% train, 15% validation, 15% test
- **Features**: 35 pre-match features (no in-match data leakage)

## 🏆 Key Features

### Non-Linear Features
- Log-rank difference with rank closeness
- Capped streaks (±5) with momentum tracking
- Career stage U-curve (peak upset potential at 50-100 matches)
- Bayesian-smoothed H2H win rates

### Advanced Features
- Nationality pair win rates (2533 pairs)
- Continent-based home advantage
- Form momentum (short-term vs long-term trends)
- Historical 3-set rates per player

### Ensemble Architecture
1. **Base Models**:
   - LightGBM (gradient boosting)
   - XGBoost (gradient boosting)
   - CatBoost (gradient boosting)

2. **Meta-Learner**: BayesianRidge (linear regression with Bayesian priors)

3. **Calibration**: Temperature Scaling for well-calibrated probabilities

## 📚 Citation

```bibtex
@software{happy_badminton_2026,
  title={Happy Badminton - AI Match Prediction},
  author={OWENLEE},
  year={2026},
  url={https://huggingface.co/spaces/owenlee-5678/happy-badminton}
}
```

## 📄 License

MIT License - see LICENSE file for details

---

**Built with ❤️ for the badminton community**
