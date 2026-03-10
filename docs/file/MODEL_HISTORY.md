# 模型演进历史

**项目**: Happy-Badminton 羽毛球比赛预测系统
**更新日期**: 2026-03-07

---

## 一、性能演变总览

| 版本 | 特征集 | LogLoss | AUC | Brier Score | Accuracy | 主要改进 |
|------|--------|---------|-----|-------------|----------|----------|
| **Baseline** | 基础特征 | 0.4793 | 0.8118 | 0.1581 | - | 初始版本 |
| **Optimized V1** | MOV Elo + 动量 + H2H + 疲劳 | 0.4071 | 0.8751 | 0.1312 | 0.8047 | 高级特征工程 |
| **Next Gen V2 (SOTA)** | **Glicko-2 + 关键分 + 波动性 + 对手质量** | **0.2329** | **0.9673** | **0.0668** | **0.9076** | **新一代特征** |
| **SimplifiedEnsemble V2** | **26 Phase-1 预赛前特征** | **0.4527** | **0.8349** | **0.1485** | - | **BayesianRidge + TemperatureScaler** |

### 性能提升对比 (NextGen V2 vs Baseline)

```
相比 Baseline:
  LogLoss:  -51.5% (0.4793 → 0.2329)
  AUC:      +19.2% (0.8118 → 0.9673)
  Brier:    -57.7% (0.1581 → 0.0668)

相比 Optimized V1:
  LogLoss:  -42.8% (0.4071 → 0.2329)
  AUC:      +10.5% (0.8751 → 0.9673)
  Brier:    -49.1% (0.1312 → 0.0668)
  Accuracy: +12.8% (0.8047 → 0.9076)
```

---

## 二、版本 0: Baseline

### 特征
- 基础解析特征（比分、局数）
- 排名差距
- 比赛级别
- 性别标记

### 模型
- 单一 LightGBM
- 默认参数

### 性能
- LogLoss: 0.4793
- AUC: 0.8118

---

## 三、版本 1: Optimized V1 (SOTA Ensemble)

### 新增特征

#### 1. MOV (Margin of Victory) 加权 Elo
**文件**: `src/data/advanced_features.py::MOVEloRating`

**核心算法**:
```python
K_effective = K_base × MOV_multiplier × level_weight
MOV_multiplier = 1 + (point_diff / total_points) × set_bonus
```

**改进点**:
- 大比分获胜 → 更多 Elo 变化
- 考虑比赛级别权重 (OG=1.5, IC=1.0)
- 考虑局数 (3局完胜 vs 2局险胜)

#### 2. 动量特征 (MomentumFeatures)
- 近期状态特征（5/10/20 场胜率）
- 连胜/连败特征
- 状态变化趋势

#### 3. H2H (Head-to-Head) 特征
- 历史交锋胜率
- 历史交锋场次
- 近期交锋状态

#### 4. 疲劳累积特征 (FatigueFeatures)
- 指数衰减疲劳计算
- 背靠背比赛检测
- 生涯中断标记

### 模型架构
```
Stacking Ensemble + Isotonic Calibration

Level 1: CatBoost + XGBoost + LightGBM
    ↓
Level 2: Logistic Regression
    ↓
Calibration: Isotonic Regression
```

### 超参数 (Optuna 优化)
```python
BEST_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.053,
    'num_leaves': 95,
    'min_data_in_leaf': 90,
    'lambda_l1': 1.34,
    'lambda_l2': 0.80,
    'feature_fraction': 0.60,
    'bagging_fraction': 0.99,
}
```

### 性能
- LogLoss: 0.4071 (-15.1% vs Baseline)
- AUC: 0.8751 (+7.8% vs Baseline)
- Brier Score: 0.1312 (-17.0% vs Baseline)
- Accuracy: 0.8047

### Ensemble 权重
- LightGBM: 2.51
- XGBoost: 2.02
- CatBoost: 1.71

### 特征重要性 Top 10
| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | total_player_matches | 1409 |
| 2 | log_rank_diff | 1169 |
| 3 | loser_fatigue | 1138 |
| 4 | winner_fatigue | 1066 |
| 5 | form_diff_20 | 942 |
| 6 | fatigue_diff | 840 |
| 7 | total_points | 495 |
| 8 | streak_diff | 482 |
| 9 | point_diff_set1 | 443 |
| 10 | duration_z | 442 |

---

## 四、版本 2: Next Gen V2 (当前 SOTA)

### 新增特征

#### 1. Glicko-2 评分系统
**文件**: `src/data/next_gen_features.py::Glicko2Rating`

**原理**: 改进自 Elo，引入 RD (Rating Deviation) 参数
- RD 反映评分的不确定性
- 新球员 RD 高 (350)，随着比赛增多 RD 降低 (最低 30)
- 长时间未比赛 RD 增加
- 引入 σ (volatility) 捕捉评分波动性

**特征**:
- `winner_g2_rating`, `loser_g2_rating` - Glicko-2 评分
- `winner_g2_rd`, `loser_g2_rd` - 评分不确定性
- `winner_g2_vol`, `loser_g2_vol` - 波动性
- `g2_rating_diff` - 评分差
- `g2_rd_sum` - 总不确定性

#### 2. 关键分能力 (ClutchPerformanceFeatures)
- `clutch_win_rate` - 接近比赛 (总分>42) 胜率
- `decider_win_rate` - 决胜局 (第3局) 胜率
- `comeback_rate` - 丢首局后逆转胜率
- `big_win_rate` - 大比分完胜率

#### 3. 实力波动 (VolatilityFeatures)
- `volatility_5` - 最近5场胜率标准差
- `volatility_10` - 最近10场胜率标准差
- `volatility_20` - 最近20场胜率标准差

#### 4. 对手质量 (OpponentQualityFeatures)
- `winner_opp_avg_rank` - 胜者最近对手平均排名
- `loser_opp_avg_rank` - 败者最近对手平均排名
- `opp_quality_diff` - 对手质量差

### 模型架构
与 V1 相同，但使用了更多特征（59 个 vs 约 40 个）

### 性能
- LogLoss: **0.2329** (-42.8% vs V1)
- AUC: **0.9673** (+10.5% vs V1)
- Brier Score: **0.0668** (-49.1% vs V1)
- Accuracy: **0.9076** (+12.8% vs V1)

---

## 五、版本 3: SimplifiedEnsemble V2 (通用预测器)

### 设计目标

无需球员数据库，用户赛前手动输入 26 个特征即可预测。

### 特征 (26 个 Phase-1 预赛前特征)

| 类别 | 特征 |
|------|------|
| 排名 | log_rank_diff, rank_closeness |
| 比赛环境 | category_flag, level_numeric, round_stage, match_month |
| 主场 | winner_home, loser_home, level_x_home, home_x_closeness |
| 近期状态 | winner_form_5, loser_form_5, form_diff_5, form_diff_10, form_diff_20 |
| 动量 | form_momentum_w, form_momentum_l, momentum_diff |
| 连胜 | streak_capped_w, streak_capped_l, streak_capped_diff |
| H2H | h2h_win_rate_bayes (贝叶斯平滑) |
| 经验 | career_stage (U 型曲线) |
| 交互 | rank_x_form_diff, rank_closeness_x_h2h, gender_x_rank |

### 模型架构
```
Stacking Ensemble + Temperature Scaling

Level 1: LightGBM + XGBoost + CatBoost
    ↓
Level 2: BayesianRidge (输出连续值，clip 到 [0,1])
    ↓
Calibration: TemperatureScaler (T ≈ 0.99)
```

### 性能
- LogLoss: **0.4527**
- AUC: **0.8349**
- Brier Score: **0.1485** (比旧版大幅改善)

### 关键改进
- 元模型从 LogisticRegression 换为 BayesianRidge
- 校准从 IsotonicRegression 换为 TemperatureScaler
- 特征从 16 个扩展到 26 个 (加入 Phase-1 非线性和交互特征)
- 新增 career_stage U 型曲线、rank_closeness、Bayesian H2H 平滑

---

## 六、关键发现

### 1. Glicko-2 的 RD 参数非常有效
- 评分不确定性提供了额外信息
- 比单纯 Elo 评分更能反映球员真实状态

### 2. 关键分能力区分度高
- 优秀球员在关键时刻表现更稳定
- `decider_win_rate` 是强特征

### 3. 波动性特征捕捉状态
- 状态不稳定的球员更难以预测
- 波动性特征帮助模型识别这类情况

### 4. 对手质量很重要
- 赛程难度影响胜负
- 连续强敌 vs 连续弱旅的表现差异

---

## 七、研究参考 (SOTA 文献引用)

### 羽毛球预测研究

| 研究 | 来源 | 年份 | 核心发现 |
|------|------|------|----------|
| ML + Technical Action Frequencies | [Nature Scientific Reports](https://www.nature.com/articles/s41598-025-87610-7) | 2025 | 结合技术动作频率提升预测 |
| Dynamic Elo Model | [ResearchGate](https://www.researchgate.net/publication/381185834_Badminton_player_ranking_model_based_on_dynamic_ELO_model) | 2024 | 动态Elo更准确反映球员实力 |
| Sequential Probability Ratio Test | [Springer](https://link.springer.com/article/10.1186/s13102-025-01078-6) | 2025 | 序列概率预测模型 |
| Stroke Forecasting | [ACM/IJCAI](https://dl.acm.org/doi/10.24963/ijcai.2024/1042) | 2024 | 球员级别预测数据集 |

### 梯度提升算法对比

| 研究 | 来源 | 年份 | 结论 |
|------|------|------|------|
| CatBoost vs LightGBM vs XGBoost | [Springer](https://link.springer.com/article/s13042-025-02654-5) | 2025 | CatBoost 在多个任务中表现最优 (R²=0.9998) |
| GBM 算法基准测试 | [arXiv](https://arxiv.org/pdf/2305.17094) | 2023 | 系统性对比四种GBM算法 |
| Football Match Prediction | [OpenAccess](https://openaccess-api.cms-conferences.org/articles/download/978-1-95865137-7_9) | 2024 | GBM应用于足球比赛预测 |

### Elo 评分改进

| 研究 | 来源 | 年份 | 核心方法 |
|------|------|------|----------|
| Tennis Elo Comparison | [ResearchGate](https://www.researchgate.net/publication/375589629_A_comparative_evaluation_of_Elo_ratings) | 2024 | 加权Elo和MOV方法 |
| Elo Reliability Study | [arXiv](https://arxiv.org/pdf/2502.10985) | 2025 | MOV方法优于标准Elo |
| Momentum Tracking | [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3685088.3685181) | 2024 | 动量特征提升预测准确性 |

### 概率校准研究

| 研究 | 来源 | 年份 | 核心发现 |
|------|------|------|----------|
| ML for Sports Betting | [arXiv](https://arxiv.org/abs/2303.06021) | 2023 | 使用校准而非准确率作为模型选择基础 |
| AI Model Calibration | [Sports-AI.dev](https://www.sports-ai.dev/blog/ai-model-calibration-brier-score) | 2024 | Platt Scaling, Isotonic Regression |

---

## 八、未来改进方向

### 短期 (可选优化)
1. 更多 Optuna trials (当前 30，可尝试 100+)
2. 不同的元模型 (如 Neural Network)
3. 交叉验证评估稳定性

### 中期 (1-2月)
1. 更多技术统计特征 (如: 杀球率、失误率)
2. 实时预测 API
3. 球员风格聚类

### 长期 (3-6月)
1. 深度学习模型 (LSTM/Transformer)
2. 计算机视觉分析 (球员姿态识别)
3. 多任务学习 (比分预测 + 获胜预测)

---

## 九、运行命令

```bash
# 训练第二代模型 (SOTA)
python scripts/train_next_gen.py

# 查看结果
cat models/next_gen_results.json

# 预测比赛
python scripts/predict.py --player1 "安赛龙" --player2 "李宗伟" --type MS
```

---

## 十、参考文献

1. Nature Scientific Reports. "Predicting badminton outcomes through machine learning and technical action frequencies." 2025.
2. Ma, X. et al. "Badminton player ranking model based on dynamic ELO model." ResearchGate, 2024.
3. Springer. "Sequential winning-percentage prediction model using expert system sequential probability ratio test." 2025.
4. ACM/IJCAI. "Benchmarking stroke forecasting with stroke-level badminton dataset." 2024.
5. Springer. "Comparative analysis of CatBoost, LightGBM, XGBoost." 2025.
6. arXiv. "Benchmarking state-of-the-art gradient boosting algorithms." 2023.
7. ResearchGate. "A comparative evaluation of Elo ratings and machine learning-based methods for tennis match result prediction." 2024.
8. Glickman, M. (1999). "The Glicko System"
9. Glickman, M. (2012). "Example of the Glicko-2 System"
