"""
Project-wide constants for feature management.

Single source of truth for leak features and post-match features.
Import from here instead of redefining in each script.
"""

# Features excluded because the training target y = (mov_elo_diff > 0).
# Including these would directly encode the target, making the model trivially accurate.
# Note: winner_elo / loser_elo / elo_diff from raw Excel are pre-match values and CAN be used.
LEAK_FEATURES: list[str] = [
    "mov_elo_diff",
    "winner_mov_elo",
    "loser_mov_elo",
]

# Features that are only known after the match ends.
# Cannot be used in real-time prediction before a match.
POST_MATCH_FEATURES: list[str] = [
    "total_points",  # total points scored during the match
    "point_diff_set1",  # first-set score differential
    "seconds_per_point",  # match pace (requires duration + total_points)
    "implied_fatigue",  # proxy = duration * sets_played
    "pace_z",  # standardized match pace
    "duration_z",  # standardized match duration
    "duration",  # raw match duration in minutes
    "sets_played",  # number of sets played
]

# Non-feature columns: identifiers, raw labels, metadata
NON_FEATURE_COLS: list[str] = [
    "winner_id",
    "loser_id",
    "match_date",
    "tournament_id",
    "type",
    "level",
    "country",
    "score",
    "round",
    "winner_assoc",
    "loser_assoc",
    "target",
    "is_retirement",
    "winner_rank",  # official ranking at match time (use log_rank_diff instead)
    "loser_rank",  # official ranking at match time (use log_rank_diff instead)
]

# Best hyperparameters found via Optuna optimization (run 2026-01).
# Pass to StackingEnsemble or individual model constructors for manual tuning.
OPTUNA_BEST_PARAMS: dict[str, dict] = {
    "lightgbm": {
        "max_depth": 5,
        "learning_rate": 0.053,
        "num_leaves": 95,
        "min_data_in_leaf": 90,
        "lambda_l1": 1.34,
        "lambda_l2": 0.80,
        "feature_fraction": 0.60,
        "bagging_fraction": 0.99,
        "n_estimators": 500,
    },
    "xgboost": {
        "max_depth": 5,
        "learning_rate": 0.053,
        "n_estimators": 500,
    },
    "catboost": {
        "depth": 5,
        "learning_rate": 0.053,
        "iterations": 500,
    },
}

# Continent mapping for nationality and venue features.
# Covers all unique player associations and tournament host countries in the dataset.
CONTINENT_MAP: dict[str, str] = {
    # Asia
    "Afghanistan": "Asia",
    "Azerbaijan": "Asia",
    "Bahrain": "Asia",
    "Bangladesh": "Asia",
    "Brunei": "Asia",
    "China": "Asia",
    "Chinese Taipei": "Asia",
    "Hong Kong": "Asia",
    "India": "Asia",
    "Indonesia": "Asia",
    "Iran": "Asia",
    "Iraq": "Asia",
    "Japan": "Asia",
    "Jordan": "Asia",
    "Kazakhstan": "Asia",
    "Korea": "Asia",
    "Kyrgyzstan": "Asia",
    "Lao": "Asia",
    "Macau": "Asia",
    "Macau China": "Asia",
    "Malaysia": "Asia",
    "Maldives": "Asia",
    "Mongolia": "Asia",
    "Myanmar": "Asia",
    "Nepal": "Asia",
    "North Korea": "Asia",
    "Pakistan": "Asia",
    "Philippines": "Asia",
    "Saudi Arabia": "Asia",
    "Singapore": "Asia",
    "Sri Lanka": "Asia",
    "Syrian Arab Republic": "Asia",
    "Tajikistan": "Asia",
    "Thailand": "Asia",
    "Timor-Leste": "Asia",
    "United Arab Emirates": "Asia",
    "Uzbekistan": "Asia",
    "Vietnam": "Asia",
    # Europe
    "Armenia": "Europe",
    "Austria": "Europe",
    "Belgium": "Europe",
    "Bulgaria": "Europe",
    "Croatia": "Europe",
    "Cyprus": "Europe",
    "Czech Republic": "Europe",
    "Czechia": "Europe",
    "Denmark": "Europe",
    "England": "Europe",
    "Estonia": "Europe",
    "Faroe Islands": "Europe",
    "Finland": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Greece": "Europe",
    "Hungary": "Europe",
    "Iceland": "Europe",
    "Ireland": "Europe",
    "Israel": "Europe",
    "Italy": "Europe",
    "Latvia": "Europe",
    "Lithuania": "Europe",
    "Luxembourg": "Europe",
    "Malta": "Europe",
    "Moldova": "Europe",
    "Netherlands": "Europe",
    "North Macedonia": "Europe",
    "Norway": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Romania": "Europe",
    "Russia": "Europe",
    "Scotland": "Europe",
    "Serbia": "Europe",
    "Slovakia": "Europe",
    "Slovenia": "Europe",
    "Spain": "Europe",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "Turkey": "Europe",
    "Ukraine": "Europe",
    "Wales": "Europe",
    # Africa
    "Algeria": "Africa",
    "Benin": "Africa",
    "Botswana": "Africa",
    "Cameroon": "Africa",
    "Central African Republic": "Africa",
    "Democratic Republic of the Congo": "Africa",
    "Egypt": "Africa",
    "Equatorial Guinea": "Africa",
    "Eritrea": "Africa",
    "Ghana": "Africa",
    "Kenya": "Africa",
    "Madagascar": "Africa",
    "Mauritius": "Africa",
    "Mayotte": "Africa",
    "Morocco": "Africa",
    "Nigeria": "Africa",
    "Réunion": "Africa",
    "Rwanda": "Africa",
    "Sierra Leone": "Africa",
    "South Africa": "Africa",
    "Tunisia": "Africa",
    "Uganda": "Africa",
    "Zambia": "Africa",
    "Zimbabwe": "Africa",
    # Americas
    "Argentina": "Americas",
    "Aruba": "Americas",
    "Barbados": "Americas",
    "Bolivia": "Americas",
    "Brazil": "Americas",
    "Canada": "Americas",
    "Chile": "Americas",
    "Colombia": "Americas",
    "Costa Rica": "Americas",
    "Cuba": "Americas",
    "Dominican Republic": "Americas",
    "Ecuador": "Americas",
    "El Salvador": "Americas",
    "Falkland Islands": "Americas",
    "French Guiana": "Americas",
    "Greenland": "Americas",
    "Guatemala": "Americas",
    "Guyana": "Americas",
    "Jamaica": "Americas",
    "Mexico": "Americas",
    "Panama": "Americas",
    "Paraguay": "Americas",
    "Peru": "Americas",
    "Puerto Rico": "Americas",
    "Suriname": "Americas",
    "Toronto": "Americas",  # data entry error (city), treated as Canada
    "Trinidad and Tobago": "Americas",
    "United States": "Americas",
    "Venezuela": "Americas",
    # Oceania
    "Australia": "Oceania",
    "Cook Islands": "Oceania",
    "Fiji": "Oceania",
    "French Polynesia (Tahiti)": "Oceania",
    "Guam": "Oceania",
    "New Zealand": "Oceania",
    "Northern Mariana Islands": "Oceania",
    "Northern Marianas": "Oceania",
}

# Tournament level mapping (higher = more prestigious).
# Values match those used in feature_engineering.py and frontend/app.py.
LEVEL_MAP: dict[str, int] = {
    "OG": 10,  # Olympic Games
    "WC": 9,  # World Championships
    "WTF": 8,  # World Tour Finals
    "S1000": 7,  # Super 1000
    "S750": 6,  # Super 750
    "S500": 5,  # Super 500
    "S300": 4,  # Super 300
    "S100": 3,  # Super 100
    "IS": 2,  # International Series
    "IC": 1,  # International Challenge
}
