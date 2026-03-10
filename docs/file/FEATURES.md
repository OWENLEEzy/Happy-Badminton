# 特征工程文档

**项目**: Happy-Badminton 羽毛球比赛预测系统
**最后更新**: 2026-03-07

---

## 零、模型特征概况

| 模型 | 特征数 | 特征类型 | 文件 |
|------|--------|----------|------|
| NextGenEnsemble | ~52 | 全量（包括 Glicko-2、关键分等） | models/next_gen_results.json |
| SimplifiedEnsemble | 26 | 仅预赛前可手动输入 | models/simplified_results.json |

---

## 一、SimplifiedEnsemble 特征 (26 个 Phase-1 预赛前特征)

用户可在赛前手动输入，无需球员数据库。由 `/api/predict-general` 端点使用。

### 1.1 排名特征

| 特征 | 说明 | 范围 |
|------|------|------|
| `log_rank_diff` | log(P1排名) - log(P2排名)，负值表示P1排名更高 | 约 -5 ~ +5 |
| `rank_closeness` | 1 / (1 + \|log_rank_diff\|)，1=势均力敌 | 0 ~ 1 |

### 1.2 比赛环境特征

| 特征 | 说明 | 范围 |
|------|------|------|
| `category_flag` | WS=1, 其他=0 | 0 或 1 |
| `level_numeric` | OG=10, WC=9, WTF=8, S1000=7, S750=6, S500=5, S300=4, S100=3, IS=2, IC=1 | 1 ~ 10 |
| `round_stage` | Q-Round1=0, Q-Round2=1, Q-Round3/4=2, Q-QF/R1=3, R2=4, R3/4=5, QF=6, SF=7, Final=8 | 0 ~ 8 |
| `match_month` | 比赛月份 (1=1月, 12=12月) | 1 ~ 12 |

### 1.3 主场特征

| 特征 | 说明 |
|------|------|
| `winner_home` | P1国籍 == 主办国 (1/0) |
| `loser_home` | P2国籍 == 主办国 (1/0) |
| `level_x_home` | level_numeric × winner_home |
| `home_x_closeness` | winner_home × rank_closeness |

### 1.4 近期状态特征

| 特征 | 说明 |
|------|------|
| `winner_form_5` | P1近5场胜率 (0~1) |
| `loser_form_5` | P2近5场胜率 (0~1) |
| `form_diff_5` | winner_form_5 - loser_form_5 |
| `form_diff_10` | 10场胜率差 |
| `form_diff_20` | 20场胜率差 |

### 1.5 动量特征

| 特征 | 说明 |
|------|------|
| `form_momentum_w` | P1的 form_5 - form_10 (上升趋势>0) |
| `form_momentum_l` | P2的 form_5 - form_10 |
| `momentum_diff` | form_momentum_w - form_momentum_l |

### 1.6 连胜特征 (非线性，封顶5)

| 特征 | 说明 |
|------|------|
| `streak_capped_w` | P1连胜 (正=连胜, 负=连败, \|x\|封顶5) |
| `streak_capped_l` | P2连胜 |
| `streak_capped_diff` | streak_capped_w - streak_capped_l |

### 1.7 H2H 特征 (贝叶斯平滑)

| 特征 | 说明 |
|------|------|
| `h2h_win_rate_bayes` | (P1胜场 + 2.5) / (总场次 + 5)，先验=5 |

### 1.8 经验特征

| 特征 | 说明 | U型曲线 |
|------|------|---------|
| `career_stage` | P1生涯总场次分箱 | ≤20→0.0, ≤50→1.0, ≤100→2.0, ≤200→1.5, ≤500→0.5, >500→0.0 |

### 1.9 交互特征

| 特征 | 说明 |
|------|------|
| `rank_x_form_diff` | log_rank_diff × form_diff_10 |
| `rank_closeness_x_h2h` | rank_closeness × (h2h_win_rate_bayes - 0.5) |
| `gender_x_rank` | category_flag × log_rank_diff |

---

## 二、特征总览 (NextGenEnsemble)

当前 NextGen 模型使用约 **52 个特征**，分为 6 大类。

| 类别 | 特征数量 | 说明 |
|------|----------|------|
| 实力评分 | 10 | MOV Elo, Glicko-2, 排名 |
| 近期状态 | 12 | 胜率、连胜、波动 |
| 历史交锋 | 3 | H2H 记录 |
| 比赛能力 | 4 | 关键分、决胜局 |
| 身体状态 | 4 | 疲劳、休息 |
| 比赛环境 | 6 | 级别、轮次、对手质量 |
| 基础特征 | 20 | 比分、时长、节奏等 |

---

## 二、实力评分特征

### 2.1 MOV Elo (Margin of Victory 加权 Elo)

**文件**: `src/data/advanced_features.py::MOVEloRating`

**原理**: 大比分获胜应该获得更多 Elo 变化

**核心公式**:
```python
K_effective = K_base × MOV_multiplier × level_weight

MOV_multiplier = 1 + (point_diff / total_points) × set_bonus
# set_bonus: 3局胜 = 1.2, 2局胜 = 1.0
```

**比赛级别权重**:
| 级别 | 权重 |
|------|------|
| OG (奥运会) | 1.5 |
| WC (世锦赛) | 1.4 |
| WTF (总决赛) | 1.3 |
| S1000 | 1.25 |
| S750 | 1.2 |
| S500 | 1.15 |
| IC/IS | 1.0 |

**特征**:
- `winner_mov_elo` - 胜者 MOV Elo
- `loser_mov_elo` - 败者 MOV Elo
- `mov_elo_diff` - Elo 差距

**预期收益**: LogLoss -0.01 ~ -0.02

### 2.2 Glicko-2 评分系统

**文件**: `src/data/next_gen_features.py::Glicko2Rating`

**原理**: 改进自 Elo，引入 RD (Rating Deviation) 参数

**关键参数**:
```python
INITIAL_RATING = 1500
INITIAL_RD = 350        # 初始不确定性
MIN_RD = 30             # 最低不确定性
TAU = 0.5              # 限制波动性变化
```

**特征**:
- `winner_g2_rating`, `loser_g2_rating` - Glicko-2 评分
- `winner_g2_rd`, `loser_g2_rd` - 评分不确定性
- `winner_g2_vol`, `loser_g2_vol` - 波动性
- `g2_rating_diff` - 评分差
- `g2_rd_sum` - 总不确定性

**预期收益**: LogLoss -0.01 ~ -0.02

### 2.3 排名特征

| 特征 | 公式 | 说明 |
|------|------|------|
| `log_rank_diff` | log(winner_rank) - log(loser_rank) | 对数排名差 |
| `level_numeric` | OG=10, WC=9 ... IC=1 | 比赛级别数值化 |

---

## 三、近期状态特征

### 3.1 滚动胜率 (MomentumFeatures)

**文件**: `src/data/advanced_features.py::MomentumFeatures`

**特征**:
| 特征 | 窗口 | 说明 |
|------|------|------|
| `winner_form_5` | 5场 | 胜者近期胜率 |
| `loser_form_5` | 5场 | 败者近期胜率 |
| `form_diff_5` | 5场 | 胜率差 |
| `form_diff_10` | 10场 | 10场胜率差 |
| `form_diff_20` | 20场 | 20场胜率差 |

**防泄露**: 使用 `shift(1)` 确保只用历史数据

### 3.2 连胜特征

| 特征 | 说明 |
|------|------|
| `winner_streak` | 胜者当前连胜场数 |
| `loser_streak` | 败者当前连败场数 (负数) |
| `streak_diff` | 连胜差值 |

### 3.3 实力波动 (VolatilityFeatures)

**文件**: `src/data/next_gen_features.py::VolatilityFeatures`

**特征**:
| 特征 | 说明 |
|------|------|
| `volatility_5` | 最近5场胜率标准差 |
| `volatility_10` | 最近10场胜率标准差 |
| `volatility_20` | 最近20场胜率标准差 |

**预期收益**: LogLoss -0.005 ~ -0.01

---

## 四、历史交锋特征 (H2H)

**文件**: `src/data/advanced_features.py::HeadToHeadFeatures`

| 特征 | 说明 |
|------|------|
| `h2h_win_rate` | 胜者对败者的历史胜率 |
| `h2h_matches` | 历史交锋场次 |
| `h2h_recent_form` | 近期交锋状态 |

**预期收益**: LogLoss -0.005 ~ -0.01

---

## 五、比赛能力特征

### 5.1 关键分能力 (ClutchPerformanceFeatures)

**文件**: `src/data/next_gen_features.py::ClutchPerformanceFeatures`

**特征**:
| 特征 | 说明 |
|------|------|
| `clutch_win_rate` | 接近比赛 (总分>42) 胜率 |
| `decider_win_rate` | 决胜局 (第3局) 胜率 |
| `comeback_rate` | 丢首局后逆转胜率 |
| `big_win_rate` | 大比分完胜率 (21-10+) |

**预期收益**: LogLoss -0.005 ~ -0.015

### 5.2 比分风格特征 (规划中)

| 特征 | 说明 |
|------|------|
| `blowout_rate` | 大比分完胜率 |
| `close_win_rate` | 险胜胜率 (21-19, 22-20) |
| `three_set_ratio` | 打满3局的比例 |

---

## 六、身体状态特征

### 6.1 疲劳累积 (FatigueFeatures)

**文件**: `src/data/advanced_features.py::FatigueFeatures`

**算法**: 指数衰减
```python
fatigue_today = fatigue_yesterday × decay^days + duration × sets / 60
# decay = 0.7 (每天衰减)
```

**特征**:
| 特征 | 说明 |
|------|------|
| `winner_fatigue` | 胜者疲劳指数 (小时) |
| `loser_fatigue` | 败者疲劳指数 |
| `fatigue_diff` | 疲劳差值 |

**关键发现**:
- 背靠背比赛时长 +15.3%
- 超过 180 天空白期: 2,156 次

### 6.2 休息特征

| 特征 | 说明 |
|------|------|
| `days_since_last_match` | 距离上场比赛天数 |
| `comeback_from_break` | 伤愈复出标记 (>30天) |

---

## 七、比赛环境特征

### 7.1 比赛重要性

| 特征 | 说明 |
|------|------|
| `round_importance` | 轮次重要性 (F=10, SF=8, QF=6...) |
| `is_knockout` | 是否淘汰赛 |
| `importance_score` | 综合重要性评分 |

### 7.2 对手质量 (OpponentQualityFeatures)

**文件**: `src/data/next_gen_features.py::OpponentQualityFeatures`

**特征**:
| 特征 | 说明 |
|------|------|
| `winner_opp_avg_rank` | 胜者最近对手平均排名 |
| `loser_opp_avg_rank` | 败者最近对手平均排名 |
| `opp_quality_diff` | 对手质量差 |

**预期收益**: LogLoss -0.003 ~ -0.008

### 7.3 主场效应 (分层分析)

**关键发现**: 主场效应因赛事级别而异

| 赛事级别 | 主场胜率 | 结论 |
|----------|----------|------|
| S1000 | 54.26% | 主场是优势 ✅ |
| S750 | 52.09% | 主场是优势 ✅ |
| S500 | 48.13% | 基本持平 |
| IS | 36.34% | **主场是劣势 ❌** |
| IC | 44.85% | 主场是劣势 ❌ |

**启示**: 需要 `Level × Home` 交互特征

---

## 八、基础解析特征

### 8.1 比分特征

| 特征 | 公式 | 说明 |
|------|------|------|
| `total_points` | sum(每局得分) | 比赛总得分 |
| `sets_played` | count(/) + 1 | 局数 (2或3) |
| `point_diff_set1` | 第1局分差 | 第一局净胜分 |

### 8.2 比赛节奏特征

| 特征 | 公式 | 说明 |
|------|------|------|
| `seconds_per_point` | Duration / Total_Points | 每分耗时 |
| `pace_z` | 标准化节奏 | Z-score by MS/WS |
| `duration_z` | 标准化时长 | Z-score by MS/WS |

### 8.3 球员特征

| 特征 | 说明 |
|------|------|
| `total_player_matches` | 球员总比赛数 |
| `is_sparse_player` | 稀疏球员标记 (≤5场) |
| `category_flag` | MS=0, WS=1 |

---

## 九、特征工程最佳实践

### 9.1 防数据泄露

```python
# 错误：使用了当前比赛信息
df['rolling_win'] = df.groupby('player_id')['won'].rolling(5).mean()

# 正确：只用历史信息
df['rolling_win'] = df.groupby('player_id')['won'].shift(1).rolling(5).mean()
```

### 9.2 标准化防泄露

```python
# 错误：在全量数据上计算统计量
mean_all = df['seconds_per_point'].mean()

# 正确：只在训练集上计算
mean_train = X_train['seconds_per_point'].mean()
X_train['pace_z'] = (X_train['seconds_per_point'] - mean_train) / std_train
X_val['pace_z'] = (X_val['seconds_per_point'] - mean_train) / std_train  # 用训练集统计量
```

### 9.3 严格禁止的泄露特征

| 特征名 | 泄露原因 |
|--------|----------|
| `elo_diff` | 直接关联胜者 |
| `winner_elo` | 胜者赛后 Elo |
| `loser_elo` | 败者赛后 Elo |
| `mov_elo_diff` | MOV Elo 差，直接关联结果 |
| `winner_rank` | 胜者赛后排名 |
| `loser_rank` | 败者赛后排名 |
| `score` | 比分直接决定胜负 |
| `sets_played` | 赛后才知道 |

---

## 十、特征重要性 (SOTA 模型)

### Top 20 特征

| 排名 | 特征 | 类别 | 重要性 |
|------|------|------|--------|
| 1 | total_player_matches | 基础 | 1409 |
| 2 | log_rank_diff | 实力 | 1169 |
| 3 | loser_fatigue | 身体 | 1138 |
| 4 | winner_fatigue | 身体 | 1066 |
| 5 | form_diff_20 | 状态 | 942 |
| 6 | fatigue_diff | 身体 | 840 |
| 7 | total_points | 基础 | 495 |
| 8 | streak_diff | 状态 | 482 |
| 9 | point_diff_set1 | 基础 | 443 |
| 10 | duration_z | 基础 | 442 |

---

## 十一、未来特征方向

### 第一批 (高优先级)
1. ✅ Glicko-2 评分系统 - 已完成
2. ✅ 关键分能力 - 已完成
3. ✅ 比赛重要性特征 - 已完成

### 第二批 (中优先级)
4. ✅ 实力波动特征 - 已完成
5. ⏳ 比分风格特征 - 规划中
6. ✅ 对手质量特征 - 已完成

### 第三批 (可选)
7. ⏳ 赛季阶段特征 - 规划中
8. ⏳ 疲劳敏感度 - 规划中
9. ⏳ 得分效率特征 - 规划中

---

## 十二、参考文献

1. Glickman, M. (1999). "The Glicko System"
2. Glickman, M. (2012). "Example of the Glicko-2 System"
3. Nature Scientific Reports (2025). "Predicting badminton outcomes through machine learning"
4. ResearchGate (2024). "Badminton player ranking model based on dynamic ELO model"
