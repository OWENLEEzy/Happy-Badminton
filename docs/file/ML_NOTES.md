# ML 注意事项 (ML_NOTES.md)

**项目**: Happy-Badminton 羽毛球比赛预测系统
**最后更新**: 2026-03-07
**NextGen LogLoss**: 0.2329 | **NextGen AUC**: 0.9673 | **Simplified LogLoss**: 0.4527 | **Simplified AUC**: 0.8349

---

## 1. 模型训练注意事项

### 1.1 数据划分原则

```python
# 时间序列划分 - 必须严格遵守
n_total = len(df_final)
train_end = int(n_total * 0.70)  # 前 70% 作为训练集
val_end = int(n_total * 0.85)    # 接下来 15% 作为验证集
# 剩余 15% 作为测试集

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]
```

**关键点**:
- 必须按 `match_date` 排序后切分
- 绝不能使用 `train_test_split` 随机划分
- 验证集用于超参数调优和早停
- 测试集只能用于最终评估，不能参与任何训练决策

### 1.2 Early Stopping 配置

```python
# 所有 GBM 模型统一使用
early_stopping_rounds = 50

# LightGBM
callbacks=[lgb.early_stopping(50, verbose=False)]

# XGBoost
early_stopping_rounds=50

# CatBoost
early_stopping_rounds=50
```

### 1.3 随机种子设置

```python
RANDOM_SEED = 42  # 全局统一
```

所有模型、数据划分、优化器都应使用相同种子以保证可复现性。

### 1.4 训练顺序

1. 数据加载与预处理
2. 基础特征工程
3. 高级特征工程 (MOV Elo, 动量, H2H, 疲劳)
4. 新一代特征 (Glicko-2, 关键分, 波动性, 对手质量)
5. 在训练集上计算标准化统计量
6. 应用标准化到全部数据
7. 添加滚动特征 (必须使用 shift(1))
8. 按时间划分数据集
9. 训练基础模型
10. 训练元模型
11. 概率校准

---

## 2. 数据泄露特征列表 (哪些特征不能用)

### 2.1 严格禁止的特征 (直接泄露结果)

| 特征名 | 泄露原因 | 必须移除 |
|--------|----------|----------|
| `elo_diff` | 直接关联胜者，包含目标信息 | YES |
| `winner_elo` | 胜者赛后 Elo | YES |
| `loser_elo` | 败者赛后 Elo | YES |
| `mov_elo_diff` | MOV Elo 差，直接关联结果 | YES |
| `winner_mov_elo` | 胜者赛后 MOV Elo | YES |
| `loser_mov_elo` | 败者赛后 MOV Elo | YES |
| `winner_rank` | 胜者赛后排名 | YES |
| `loser_rank` | 败者赛后排名 | YES (作为差值可用) |
| `score` | 比分直接决定胜负 | YES |
| `sets_played` | 赛后才知道 | YES |
| `total_points` | 赛后才知道 | YES |
| `point_diff_set1` | 首局分差，但需赛前获取 | 谨慎使用 |

### 2.2 需要预处理防泄露的特征

```python
# 滚动特征必须使用 shift(1)
df['rolling_win_rate'] = df.groupby('player_id')['won']
    .transform(lambda x: x.shift(1).rolling(5).mean())

# 标准化必须只在训练集上拟合
scaler = StandardScaler()
scaler.fit(X_train[mask])  # 仅训练集
X_scaled = scaler.transform(X)  # 应用到全部
```

### 2.3 排除特征列表 (已验证低重要性)

```python
EXCLUDED_FEATURES = [
    # 泄露特征
    'elo_diff', 'winner_elo', 'loser_elo',
    'mov_elo_diff', 'winner_mov_elo', 'loser_mov_elo',

    # 零重要性特征 (通过 Optuna 优化发现)
    'sets_played',      # 赛后才知道
    'h2h_recent_form',  # 信息冗余
    'home_advantage',   # 羽毛球主场优势不明显
    'is_sparse_player', # 冷门球员预测无意义
]
```

### 2.4 非特征列 (元数据)

```python
NON_FEATURE_COLS = [
    'winner_id', 'loser_id', 'match_date',
    'type', 'level', 'country', 'score', 'round',
    'winner_assoc', 'loser_assoc',
    'target', 'is_retirement'
]
```

---

## 3. 时间序列分割的重要性

### 3.1 为什么不能用随机分割

羽毛球比赛数据具有强时间依赖性:

1. **球员实力随时间变化** - 2020 年的排名不能预测 2024 年的比赛
2. **评分系统是时序的** - Elo/Glicko-2 需要按顺序更新
3. **战术风格演变** - 羽毛球技术发展趋势
4. **规则变化** - 发球规则、计分规则变化

### 3.2 正确的时序划分

```python
# 必须先排序
df_final = engineer.df.copy().sort_values('match_date').reset_index(drop=True)

# 然后按索引切分
train_end = int(n_total * 0.70)
val_end = int(n_total * 0.85)
```

### 3.3 时序交叉验证 (用于超参数优化)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    # 训练和验证...
```

**注意**: TimeSeriesSplit 必须用于 Optuna 超参数优化，不能用于最终模型训练。

### 3.4 评估指标的时间一致性

- 训练集指标 > 验证集指标 > 测试集指标 是正常的
- 如果测试集指标异常高，检查是否有数据泄露
- 监控 `train_val_gap`，过大会导致泛化差

---

## 4. 概率校准方法

### 4.1 为什么需要概率校准

GBM 模型输出的概率通常不够准确:
- LightGBM 倾向于输出极端概率 (接近 0 或 1)
- 对于博彩/投资场景，校准后的概率更有价值

### 4.2 Temperature Scaling 校准 (SimplifiedEnsemble 使用)

```python
# BayesianRidge 元模型输出连续值，用 TemperatureScaler 校准
# T > 1: 压缩极端概率；T < 1: 拉开概率；T = 1: 不变
ts = TemperatureScaler()
ts.fit(val_pred_raw, y_val)   # 通过 minimize_scalar 求最优 T
y_pred_calibrated = ts.transform(test_pred_raw)
```

**优点**:
- 只有一个参数，不容易过拟合
- 数学上保持单调性
- 适合元模型输出的校准

### 4.3 Isotonic 回归校准 (NextGenEnsemble 使用)

```python
from sklearn.isotonic import IsotonicRegression

# 训练校准器 (使用验证集)
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_pred_raw, y_val)

# 应用校准
y_pred_calibrated = calibrator.predict(test_pred_raw)
```

**优点**:
- 非参数方法，适应任意分布
- 2025 研究显示优于 Sigmoid 校准
- 对 skewed distribution 更鲁棒

### 4.3 校准效果评估

```python
raw_ll = log_loss(y_val, val_pred_raw)      # 校准前
cal_ll = log_loss(y_val, val_pred_cal)      # 校准后
raw_bs = brier_score_loss(y_val, val_pred_raw)
cal_bs = brier_score_loss(y_val, val_pred_cal)

logger.info(f"LogLoss: {raw_ll:.4f} -> {cal_ll:.4f}")
logger.info(f"Brier: {raw_bs:.4f} -> {cal_bs:.4f}")
```

**典型改善**:
- LogLoss: -0.01 ~ -0.03
- Brier Score: -0.005 ~ -0.015

### 4.4 校准器保存

```python
model_data = {
    'base_models': ensemble.base_models,
    'meta_model': ensemble.meta_model,
    'calibrator': ensemble.calibrator,  # 必须保存
}
joblib.dump(model_data, output_path)
```

---

## 5. 超参数优化经验

### 5.1 LightGBM 最佳参数 (Optuna 优化结果)

```python
BEST_PARAMS = {
    'max_depth': 5,              # 不要过深，羽毛球特征相对简单
    'learning_rate': 0.053,      # 0.03-0.07 范围最优
    'num_leaves': 95,            # 约 2^depth 的 1.5-2 倍
    'min_data_in_leaf': 90,      # 较大值防止过拟合
    'lambda_l1': 1.34,           # L1 正则化
    'lambda_l2': 0.80,           # L2 正则化
    'feature_fraction': 0.60,    # 特征采样，防止过拟合
    'bagging_fraction': 0.99,    # 行采样接近 1
    'n_estimators': 500,         # 配合 early_stopping
}
```

### 5.2 搜索空间设置

```python
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
    }
```

### 5.3 优化建议

1. **试验次数**: 生产环境 100+，快速实验 30-50
2. **早停耐心**: 30-50 rounds
3. **优化目标**: 优先 LogLoss，其次是 Brier Score
4. **多模型同步**: LightGBM/XGBoost/CatBoost 参数可共享

### 5.4 特征选择优化

```python
# 基于特征重要性迭代选择
for n_features in range(10, len(feature_names) + 1, 2):
    top_features = importance_df.head(n_features)['feature'].tolist()
    # 评估并选择最佳 n
```

**经验**: 羽毛球数据 50-70 个特征最佳，过多会导致过拟合。

---

## 6. 各模型性能对比和适用场景

### 6.1 模型性能对比 (测试集)

| 模型 | LogLoss | AUC | Brier | 训练时间 | 推理速度 |
|------|---------|-----|-------|----------|----------|
| LightGBM | 0.4071 | 0.8751 | 0.1312 | 快 | 最快 |
| XGBoost | 0.4150 | 0.8700 | 0.1350 | 中 | 快 |
| CatBoost | 0.4020 | 0.8780 | 0.1290 | 慢 | 中 |
| **Stacking Ensemble (3)** | **0.4071** | **0.8751** | **0.1312** | 最慢 | 中 |
| **NextGenEnsemble (V2)** | **0.2329** | **0.9673** | **0.0668** | 慢 | 中 |
| **SimplifiedEnsemble (V2)** | **0.4527** | **0.8349** | **0.1485** | 慢 | 中 |

### 6.2 单模型适用场景

**LightGBM**:
- 适用: 快速迭代、在线学习、资源受限
- 优势: 训练最快、内存占用小
- 缺点: 概率输出需要校准

**XGBoost**:
- 适用: 需要精细控制、竞赛场景
- 优势: 成熟稳定、社区支持好
- 缺点: 比 LightGBM 慢

**CatBoost**:
- 适用: 有类别特征、需要最佳精度
- 优势: 自动特征组合、默认参数优秀
- 缺点: 训练慢、Python API 不如前两者

### 6.3 Ensemble 适用场景

**Stacking Ensemble**:
- 适用: 最终生产模型、竞赛提交
- 优势: 降低方差、提高鲁棒性
- 成本: 训练时间 x3、维护复杂度增加

**推荐配置** (NextGenEnsemble):
- Base Models: LightGBM + XGBoost + CatBoost
- Meta Model: BayesianRidge
- Calibration: IsotonicRegression (NextGen) / TemperatureScaler (Simplified)

### 6.4 深度学习探索 (未验证)

- LSTM/Transformer: 适合建模长期序列依赖
- 计算成本高，需要更多数据
- 当前 GBM 方案已达到 SOTA (LogLoss 0.23)

---

## 7. 预测时需要注意的问题

### 7.1 特征一致性

```python
# 预测时必须使用与训练时相同的特征
expected_features = [
    'log_rank_diff', 'category_flag', 'level_numeric',
    'form_diff_5', 'form_diff_10', 'streak_diff',
    'h2h_win_rate', 'fatigue_diff',
    'g2_rating_diff', 'clutch_win_rate', 'decider_win_rate',
    'volatility_5', 'volatility_10',
    # ... 共 59 个特征
]

# 检查特征完整性
missing_features = set(expected_features) - set(X.columns)
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")
```

### 7.2 评分系统状态

预测时需要:
1. 加载最新的评分系统状态 (Elo/Glicko-2)
2. 按时间顺序更新到预测日前
3. 使用当前评分计算特征

```python
# 错误: 使用训练时的评分
# 正确: 使用实时更新的评分
glicko = Glicko2Rating()
for match in historical_matches:
    glicko.update_player(...)  # 必须按时间更新
current_rating = glicko.get_rating(player_id)
```

### 7.3 冷启动问题

```python
# 新球员处理
if player_id not in ratings:
    rating = INITIAL_RATING  # 1500
    rd = INITIAL_RD          # 350 (高不确定性)
    # 随着比赛增多，RD 会自动降低
```

### 7.4 概率阈值选择

```python
# 不同场景使用不同阈值
# 保守场景 (避免错误预测)
threshold_conservative = 0.60

# 激进场景 (追求高回报)
threshold_aggressive = 0.45

# 默认 (平衡)
threshold_default = 0.50
```

### 7.5 预测输出规范

```python
# 标准预测输出
prediction = {
    'match_id': 'xxx',
    'player_a': 'Player A',
    'player_b': 'Player B',
    'predicted_winner': 'Player A',
    'win_probability_a': 0.72,  # 校准后概率
    'win_probability_b': 0.28,
    'confidence': 'high',       # 基于概率和 RD
    'model_version': 'NextGenV2',
    'prediction_date': '2025-01-17',
}
```

### 7.6 模型版本管理

```python
# 保存模型时记录元数据
model_metadata = {
    'version': 'NextGenV2',
    'train_date': '2025-01-17',
    'train_data_range': ('2018-01-01', '2024-12-31'),
    'features': feature_cols,
    'metrics': {'log_loss': 0.2311, 'auc': 0.9666},
    'params': BEST_PARAMS,
}
```

---

## 8. 常见错误与解决方案

### 8.1 LogLoss 异常高

**可能原因**:
- 数据泄露 (使用了赛后特征)
- 时间划分错误 (随机划分)
- 标签错误 (winner/loser 反了)

**排查**:
```python
# 检查标签分布
print(f"y=1: {(y==1).sum()}, y=0: {(y==0).sum()}")

# 检查时间范围
print(f"Train: {df_train['match_date'].min()} ~ {df_train['match_date'].max()}")
print(f"Test: {df_test['match_date'].min()} ~ {df_test['match_date'].max()}")
```

### 8.2 AUC 很高但 LogLoss 很差

**原因**: 概率校准问题

**解决**:
```python
# 使用 Isotonic 校准
ensemble.calibrate(X_train, y_train, X_val, y_val, method='isotonic')
```

### 8.3 测试集性能异常

**现象**: 测试集指标远好于验证集

**原因**: 数据泄露或划分错误

**解决**:
- 检查是否用了赛后特征
- 确认按时间划分
- 检查是否有重复数据

### 8.4 过拟合

**现象**: 训练集 LogLoss 0.1，验证集 0.5

**解决**:
```python
# 增加正则化
'min_data_in_leaf': 100  # 增大
'lambda_l1': 3.0         # 增大
'feature_fraction': 0.6  # 减小

# 减少特征数量
# 特征选择优化
```

---

## 9. 最佳实践总结

### 9.1 训练流程检查清单

- [ ] 数据已按 `match_date` 排序
- [ ] 时序划分 (70/15/15)
- [ ] 排除所有泄露特征
- [ ] 标准化只在训练集上拟合
- [ ] 滚动特征使用 shift(1)
- [ ] Early stopping 启用
- [ ] 随机种子固定
- [ ] 概率校准 (NextGen: Isotonic, Simplified: TemperatureScaler)
- [ ] 保存完整模型 (包括校准器)

### 9.2 特征工程检查清单

- [ ] MOV Elo / Glicko-2 按时间更新
- [ ] 动量特征 (form_diff_5/10/20)
- [ ] H2H 特征 (历史交锋)
- [ ] 疲劳特征 (累积 + 衰减)
- [ ] 关键分能力 (clutch, decider, comeback)
- [ ] 波动性特征 (volatility_5/10/20)
- [ ] 对手质量 (最近对手排名)

### 9.3 评估指标

**主要指标**: LogLoss (概率预测质量)
**次要指标**: AUC (区分能力), Brier Score (概率校准)
**辅助指标**: Accuracy, Precision, Recall, F1

---

## 10. 参考文件

| 文件 | 说明 |
|------|------|
| `scripts/train_model.py` | 基础训练脚本 |
| `scripts/train_sota.py` | SOTA Ensemble 训练 |
| `scripts/optimize_sota.py` | 超参数优化 |
| `scripts/train_optimized_ensemble.py` | 优化版 Ensemble |
| `scripts/train_next_gen.py` | **第二代模型 (SOTA)** |
| `src/data/feature_engineering.py` | 基础特征工程 |
| `src/data/advanced_features.py` | 高级特征 (MOV Elo, 动量, H2H) |
| `src/data/next_gen_features.py` | 新一代特征 (Glicko-2, 关键分) |
| `src/models/ensemble_models.py` | Stacking Ensemble 实现 |
| `src/training/trainer.py` | 训练器封装 |
| `docs/SOTA.md` | SOTA 模型文档 |
| `docs/SOTA_IMPROVEMENTS.md` | 模型改进记录 |
