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

- **Two Prediction Modes**: Quick (fast) or Expert (high accuracy)
- **Match Winner Prediction**: Predict which player will win with probability confidence
- **Set Count Prediction**: Forecast whether a match will go to 2 or 3 sets
- **Real-time Inference**: Fast predictions powered by pre-trained models
- **Driving Factors Analysis**: See which features most influence each prediction

## 🚀 How to Use

1. Visit the Space homepage
2. Choose prediction mode:
   - **Quick Mode**: Only need ranking, nationality, ELO, H2H, tournament info → 30 seconds
   - **Expert Mode**: Need all fields including recent form, streak, career matches → highest accuracy
3. Enter match details for both players
4. Click "Predict Winner"
5. View results with confidence interval and driving factors

## 📊 Model Performance

### Quick Mode (21 features)
- **AUC**: 0.8703
- **LogLoss**: 0.4247
- **Brier Score**: 0.1320
- **Accuracy**: 81.0%

### Expert Mode (35 features)
- **AUC**: 0.9608
- **LogLoss**: 0.2316
- **Brier Score**: 0.0722
- **Accuracy**: 89.7%

### Set Count Model
- **AUC**: 0.6635
- **LogLoss**: 0.5583

## 🧠 Model Details

Models are automatically downloaded from [HuggingFace Model Hub](https://huggingface.co/owenlee-5678/happy-badminton-models) on first launch.

**Model Files**:
- `quick_ensemble.pkl` (3.1MB) - Quick mode model (21 features)
- `simplified_ensemble.pkl` (3.3MB) - Expert mode model (35 features)
- `set_count_model.pkl` (1.1MB) - Set count prediction model
- Feature metadata and nationality pair lookup tables

## 📖 Input Data Required

**All data available from [BadmintonRanks.com](https://badmintonranks.com)**:

### How to Find Each Field
- **Rankings**: Player page → Profile tab (shows current ranking)
- **ELO Rating**: Player page → Match Details tab
- **Recent Form** (5/10/20): Player page → Match Details tab
- **Streaks**: Player page → Winning Streak tab
- **Head-to-Head (H2H)**: Player page → Head-to-Head tab
- **Career Matches**: Player page → Profile tab (shows total wins-losses, e.g., "453-81")

### Required Fields (Both Modes)
- Player names, nationality, **BWF ranking (required)**
- **ELO rating (required)** - Top 3 feature importance (9.5%)
- Tournament level, round, host country
- H2H record (wins/total)

### Additional Fields (Expert Mode Only)
- Recent form: wins in last 5/10/20 matches
- Current win/loss streak
- Career total matches
- Historical 3-set match rate

**Note**: ELO is **required** in both modes - it's a critical feature for prediction accuracy.

## 🔧 Technical Stack

- **Backend**: Flask 3.0+
- **ML**: scikit-learn, LightGBM, XGBoost, CatBoost
- **Models**: Custom stacking ensemble with BayesianRidge meta-learner
- **Calibration**: Temperature Scaling for probability calibration
- **Frontend**: Vanilla JavaScript (no framework dependencies)
- **Deployment**: Docker container with auto-download from Model Hub

## 📦 Training Data

- **Source**: BWF official tournament records (2019-2025)
- **Matches**: ~15,000 professional matches
- **Time-based split**: 70% train, 15% validation, 15% test
- **Features**: 21 (Quick) or 35 (Expert) pre-match features
- **No data leakage**: All features are pre-match only (rolling features use shift(1))

## 🏆 Key Features

### Core Features (Both Modes)
- Log-rank difference with rank closeness
- ELO rating and ELO difference
- Tournament level and round
- Home advantage (nationality and continent-based)
- Bayesian-smoothed H2H win rates
- Nationality pair win rates (2533 pairs)

### Expert Mode Additional Features
- Recent form momentum (5/10/20 match windows)
- Capped win/loss streaks (±5)
- Career stage U-curve
- Historical 3-set match rates
- Feature interactions (rank × form, rank × H2H)

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
