# CLAUDE.md

> **Project**: Happy-Badminton | 羽毛球比赛预测系统
> **Description**: 基于 Glicko-2 + GBM Stacking Ensemble（BayesianRidge 元学习）的羽毛球比赛胜负预测

---

## TOP 10 必读规则

| #   | 规则                       | 说明                                                                                         |
| --- | -------------------------- | -------------------------------------------------------------------------------------------- |
| 1   | **质量门禁**               | 见「完成检查清单」章节，完成前必须执行                                                       |
| 2   | **先读后写**               | 修改任何文件前必须先用 Read 工具读取，理解现有代码                                           |
| 3   | **时序分割**               | 数据集**必须**按时间顺序切分（70/15/15），严禁随机 split                                     |
| 4   | **防数据泄露**             | 见「红线禁区」中的 `LEAK_FEATURES` 列表，这些特征**绝对不能**用于训练                       |
| 5   | **shift(1) 原则**          | 所有 rolling 特征必须用 `shift(1)`，避免使用当局比赛信息                                     |
| 6   | **特征一致性**             | 预测时从 `models/*_results.json` 加载训练时的特征列表，顺序必须完全一致                     |
| 7   | **搜索优先**               | 写代码前用 Grep 搜索已有实现，避免重复造轮子                                                 |
| 8   | **测试覆盖**               | 新核心逻辑必须有测试，运行 `pytest` 验证                                                     |
| 9   | **禁止 push**              | Claude 可以 commit，但**禁止 push** 到远程仓库                                               |
| 10  | **代码用英文**             | 变量名、函数名、注释、docstring 全部用英文；中文仅限 Claude 回复和用户可见的输出内容         |
| 11  | **前端必填原则**           | 所有参与训练的特征在前端**必须强制填写**，不能设为 optional；训练用了什么，前端就要求填什么   |

---

## 完成检查清单（最重要！）

> 🚨 **没有完成检查就说"完成了" = 严重违规**

### 质量门禁（按顺序执行）

```bash
ruff check .          # 代码规范检查
ruff format .         # 代码格式化
pytest tests/ -v      # 运行全部测试
```

### 检查清单

| 检查项         | 命令                           | 说明            |
| -------------- | ------------------------------ | --------------- |
| **Ruff**       | `ruff check .`                 | 代码规范检查    |
| **Format**     | `ruff format .`                | 代码格式化      |
| **Test**       | `pytest tests/ -v`             | 单元测试        |

### 代码审查项

| 类别              | 检查项                                                              |
| ----------------- | ------------------------------------------------------------------- |
| **数据泄露**      | 没有使用 `LEAK_FEATURES` 中的特征                                   |
| **时序安全**      | Rolling 特征使用了 `shift(1)`，数据集按时间切分                     |
| **空值/边界**     | 空列表/空 DataFrame 不报错，索引越界有保护                          |
| **错误处理**      | 文件操作、模型加载有 `try/except`                                   |
| **清理**          | 删除调试用的 `print()` 语句                                         |

### 检查报告格式（完成后必须输出）

```
📝 改动报告：
📁 文件：xxx.py
✏️ 改了什么：...
🤔 为什么要改：...
✨ 改完之后：...

✅ 功能完成检查报告
📋 Ruff Check:  ✅ 通过
📋 Ruff Format: ✅ 通过
📋 Pytest:      ✅ 通过

📦 已提交："commit message"
```

---

## 红线禁区（绝对不能违反）

### ML 数据安全

| 禁止事项                                          | 后果                       |
| ------------------------------------------------- | -------------------------- |
| 🚫 使用以下 `LEAK_FEATURES` 训练模型               | 数据泄露，模型无法泛化     |
| 🚫 对数据集做随机 train/test split                 | 未来信息泄露到训练集       |
| 🚫 在非训练集上 fit scaler / encoder               | 数据泄露，评估结果虚高     |
| 🚫 rolling 特征不用 `shift(1)`                     | 使用了当局比赛结果         |
| 🚫 `.map(lambda x: series.iloc[-1])` 回填滚动特征   | 所有行拿到最终时点值，泄露未来 |
| 🚫 预测时不加载 `*_results.json` 特征列表          | 特征不一致导致预测错误     |
| 🚫 直接修改 `data/raw/Tournament Results.xlsx`     | 原始数据不可逆损坏         |

```python
# Data leakage features - NEVER use for training
LEAK_FEATURES = [
    'elo_diff', 'winner_elo', 'loser_elo',              # Post-match Elo
    'mov_elo_diff', 'winner_mov_elo', 'loser_mov_elo',  # MOV Elo
    'winner_rank', 'loser_rank',                         # Post-match rank
    'sets_played', 'total_points',                       # Post-match stats
    'score',                                             # Match result
]
```

### 代码质量

| 禁止事项                                 | 后果                           |
| ---------------------------------------- | ------------------------------ |
| 🚫 代码注释（`#`）或 docstring 使用中文   | 违反代码语言规范               |
| 🚫 硬编码超参数（K-factor、窗口大小等）   | 应在 `config.yaml` 中统一管理  |
| 🚫 不使用类型注解的新函数                 | 降低代码可读性和可维护性       |

### 代码提交

| 禁止事项                      | 后果                                                |
| ----------------------------- | --------------------------------------------------- |
| 🚫 未经测试就说"完成了"       | Bug 上线后才发现                                    |
| 🚫 commit message 暴露 AI 身份 | 不能有 Claude/AI/LLM/Co-Authored-By                 |
| 🚫 执行 git push              | 推送必须由用户自己操作                              |
| 🚫 使用中文 commit message    | 所有 git commit 必须使用英文                        |

**Commit 规范**：
- 英文 commit message
- 格式：`<type>: <description>`
- Type: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`
- 示例：`feat: add Glicko-2 volatility features`

---

## 行为规范

| 规则                       | 说明                                                                               |
| -------------------------- | ---------------------------------------------------------------------------------- |
| **永远用简体中文回复**     | 所有**回复**用中文；代码（注释、docstring、变量名）必须用英文                     |
| **讨论时不写代码**         | 讨论阶段只分析逻辑和方案，不实际修改代码；必须等用户确认后再动手                 |
| **改前必读文件**           | 修改任何文件前先用 Read 工具读取完整内容                                           |
| **新依赖要确认**           | 添加 Python 包前先问用户                                                           |
| **清理调试代码**           | 完成后删除 print() 语句                                                            |
| **一次只改一个功能**       | 不要顺手改其他东西                                                                 |
| **先搜再写**               | 写新函数前先 Grep 搜索，避免重复实现                                               |

**代码语言规范**：

```python
# ✅ Correct
def calculate_glicko_rating(matches: list[dict]) -> float:
    """Calculate Glicko-2 rating based on match history."""
    # Filter incomplete matches
    valid = [m for m in matches if m.get("score") is not None]
    return sum(m["rating_change"] for m in valid) / len(valid)

# ❌ Wrong
def calculate_glicko_rating(matches: list[dict]) -> float:
    """根据历史比赛计算 Glicko-2 评分。"""
    # 过滤不完整的比赛
    valid = [m for m in matches if m.get("score") is not None]
    return sum(m["rating_change"] for m in valid) / len(valid)
```

---

## Common Commands

### Development Setup
```bash
uv sync                      # Install / sync dependencies
```

### One-Click Launch (train if needed + start server)
```bash
uv run python main.py              # auto-train if model missing, then start
uv run python main.py --train      # force re-train even if model exists
uv run python main.py --port 8080  # custom port
./run.sh                           # shell shortcut (Mac/Linux)
```

### Train / Validate / Optimise (standalone)
```bash
uv run python scripts/train_simplified.py   # train SimplifiedEnsemble
uv run python scripts/validate_model.py     # evaluate on test set
uv run python scripts/optimize_sota.py      # Optuna hyperparameter search (optional)
```

### Testing & Quality
```bash
uv run ruff check .
uv run ruff format .
uv run pytest tests/ -v
```

---

## Architecture

### Data Flow
```
Raw Excel (data/raw/Tournament Results.xlsx)
    ↓
preprocess_pipeline() → Filter retirees, parse scores, handle outliers
    ↓
build_advanced_features() → MOV Elo, Form (MomentumFeatures), H2H, Fatigue
    ↓
FeatureEngineer → Standardization, Rolling features (all with shift(1))
    ↓
Time-based split (70/15/15) → Train/Val/Test
    ↓
StackingEnsemble → LightGBM + XGBoost + CatBoost (base) + BayesianRidge (meta)
    ↓
Calibration: TemperatureScaler
    ↓
predict_proba_calibrated() → Final prediction
```

### Key Modules

| 文件 | 职责 |
|------|------|
| `src/data/loader.py` | Loads Excel with 3 sheets (Matches, Tournaments, Players), merges into DataFrame with 17 columns |
| `src/data/preprocessor.py` | Filters retirements (3.9%), handles duration outliers (>150 min), parses scores |
| `src/data/advanced_features.py` | `MOVEloRating`, `MomentumFeatures`, `HeadToHeadFeatures`, `FatigueFeatures` |
| `src/models/ensemble_models.py` | `StackingEnsemble` (LGBM + XGBoost + CatBoost + BayesianRidge meta) + `TemperatureScaler` |
| `src/utils/constants.py` | Single source of truth: `LEAK_FEATURES`, `POST_MATCH_FEATURES`, `LEVEL_MAP` |
| `frontend/app.py` | Flask API: `GET /`, `POST /api/predict-general`, `GET /api/health` |
| `config.yaml` | All hyperparameters and thresholds (single source of truth) |

---

## Critical ML Rules

### Time-Series Splitting (MUST)
```python
df_final = engineer.df.sort_values('match_date').reset_index(drop=True)
train_end = int(n_total * 0.70)
val_end   = int(n_total * 0.85)
X_train = X.iloc[:train_end]
X_val   = X.iloc[train_end:val_end]
X_test  = X.iloc[val_end:]
```

### Rolling Features Must Use shift(1)
```python
# WRONG - uses current match info
df['rolling_win'] = df.groupby('player_id')['won'].rolling(5).mean()

# CORRECT - uses only past matches
df['rolling_win'] = df.groupby('player_id')['won'].shift(1).rolling(5).mean()
```

### Rolling Features Must Map Back by Row Index, NOT .iloc[-1]
When computing per-player rolling stats and mapping back to a match-level df,
**never** use `.map(lambda x: series.iloc[-1])` — this gives every match the
player's FINAL accumulated value (future info leaked to early matches).

```python
# WRONG — all matches get the player's last-ever form value
df['winner_form_5'] = df['winner_id'].map(
    lambda x: win_rates[x].iloc[-1] if x in win_rates else 0.5
)

# CORRECT — preserve orig_idx, compute via transform, reindex back
all_matches['df_idx'] = ...  # original df row index
all_matches['form_5'] = all_matches.groupby('player_id')['won'].transform(
    lambda x: x.astype(int).shift(1).rolling(5, min_periods=3).mean()
)
winner_form = all_matches[all_matches['role'] == 'winner'].set_index('df_idx')['form_5']
df['winner_form_5'] = winner_form.reindex(df.index).fillna(0.5).values
# See src/data/advanced_features.py MomentumFeatures.add_form_features()
```

### Standardization - Fit on Train Only
```python
scaler = StandardScaler()
scaler.fit(X_train)         # Train set only
X_scaled = scaler.transform(X)  # Apply to all
```

### User-facing Data Source
All player stats shown/entered in the frontend (ranking, ELO, recent form, streak, career matches,
3-set rate) are looked up from **[BadmintonRanks.com](https://badmintonranks.com)**.
Do NOT say "BWF official website" when referring to where users look up stats.
The raw training data (`data/raw/Tournament Results.xlsx`) is separately sourced from BWF match records.

### General Predictor Workflow (SimplifiedEnsemble)
1. User POSTs match params to `POST /api/predict-general`
2. `build_general_features()` computes features from user input (stats sourced from BadmintonRanks.com)
3. Feature vector aligned to `models/simplified_results.json` feature list (order matters)
4. `model.predict_proba_calibrated()` → TemperatureScaler output
5. `bootstrap_confidence_interval()` → P10–P90 model spread

---

## Model Performance Reference

| Model | LogLoss | AUC | Brier | Features | Notes |
|-------|---------|-----|-------|----------|-------|
| **SimplifiedEnsemble V2** | **0.4527** | **0.8349** | **0.1485** | 26 | Pre-match only, BayesianRidge + TemperatureScaler |

Used by `/api/predict-general`. All 26 features can be input manually — no player database required.

---

## File Structure

```
Happy-Badminton/
├── main.py                        # One-click entry point (auto-train + start server)
├── run.sh                         # Shell shortcut for main.py
├── src/
│   ├── data/
│   │   ├── loader.py              # Excel loading & merging
│   │   ├── preprocessor.py        # Cleaning & outlier handling
│   │   ├── advanced_features.py   # MOVEloRating, MomentumFeatures, H2H, Fatigue
│   │   └── feature_engineering.py # Standardization & rolling
│   ├── models/
│   │   └── ensemble_models.py     # StackingEnsemble + TemperatureScaler
│   └── utils/
│       ├── constants.py           # LEAK_FEATURES, POST_MATCH_FEATURES, LEVEL_MAP (single source)
│       ├── helpers.py
│       └── logger.py              # Loguru setup
├── scripts/
│   ├── train_simplified.py        # Train SimplifiedEnsemble (general predictor)
│   ├── validate_model.py          # Evaluate on test set (--model simplified)
│   └── optimize_sota.py           # Optuna hyperparameter search for base models
├── frontend/
│   ├── app.py                     # Flask API: GET /, POST /api/predict-general, GET /api/health
│   └── index.html                 # 4-view SPA (Home / Quick / Expert / Result)
├── models/                        # Trained model files (gitignored except .gitkeep)
│   ├── simplified_ensemble.pkl    # SimplifiedEnsemble V2 (AUC 0.8349, 26 features)
│   ├── simplified_results.json    # Feature list + metrics
│   └── simplified_feature_importance.json  # LGBM importances for driving factors
├── data/
│   └── raw/                       # Original Excel (committed, DO NOT MODIFY)
├── tests/                         # Pytest test files
├── config.yaml                    # Hyperparameters (single source of truth)
├── pyproject.toml                 # Project config + tool settings
└── CLAUDE.md                      # This file
```

---

## Config Reference

`config.yaml` 是所有超参数的唯一数据源，修改超参数必须改这里：

| 配置项 | 说明 |
|--------|------|
| Rolling windows | `[3, 5, 10]` |
| Elo K-factor | `32` |
| Cold start threshold | `5` matches |
| Random seed | `42` |
| Early stopping rounds | `50` |

---

## Quick Reference

```bash
# Quality gate (run before claiming done)
uv run ruff check . && uv run ruff format . && uv run pytest tests/ -v

# One-click launch (auto-train if needed + start server)
uv run python main.py

# Train / validate standalone
uv run python scripts/train_simplified.py
uv run python scripts/validate_model.py
```

| 操作         | Claude 权限 |
| ------------ | ----------- |
| `git commit` | 允许        |
| `git push`   | **禁止**    |
| `python *`   | 允许        |
| 修改原始数据 | **禁止**    |
