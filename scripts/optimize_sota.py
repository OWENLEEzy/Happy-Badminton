"""
SOTA 模型优化脚本

功能：
1. 特征重要性分析 (SHAP)
2. 超参数优化 (Optuna)
3. 特征选择优化

运行:
    python scripts/optimize_sota.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import loguru

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.utils.constants import SOTA_EXCLUDE
from src.data.loader import load_and_merge
from src.data.preprocessor import preprocess_pipeline
from src.data.advanced_features import build_advanced_features, get_advanced_feature_columns
from src.data.feature_engineering import FeatureEngineer

logger = setup_logger()


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_optimization_data(config: dict):
    """准备优化数据"""
    logger.info("=" * 60)
    logger.info("准备优化数据...")
    logger.info("=" * 60)

    # 加载和预处理数据
    df = load_and_merge(config["data"]["raw_path"])
    df_clean = preprocess_pipeline(df)
    df_advanced = build_advanced_features(df_clean)

    # 特征工程
    engineer = FeatureEngineer(df_advanced)
    engineer.add_basic_features()
    engineer.add_fatigue_features()

    # 标准化
    n = len(engineer.df)
    train_size = int(n * 0.7)
    train_mask = pd.Series([False] * n, index=engineer.df.index)
    train_mask.iloc[:train_size] = True

    engineer.fit_scalers(train_mask)
    engineer.apply_standardization()
    engineer.add_rolling_features()

    # 获取特征列（移除泄露特征和赛后特征）
    basic_features = engineer.get_feature_columns()
    advanced_features = get_advanced_feature_columns()

    all_features = list(dict.fromkeys(basic_features + advanced_features))
    all_features = [f for f in all_features if f in engineer.df.columns and f not in SOTA_EXCLUDE]

    # 准备数据
    df_final = engineer.df.copy()
    df_final = df_final.sort_values("match_date").reset_index(drop=True)

    y = (df_final["mov_elo_diff"] > 0).astype(int).values
    X = df_final[all_features].copy().fillna(0)

    # 数据划分
    n_total = len(df_final)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y[val_end:]

    logger.info(f"特征数: {len(all_features)}")
    logger.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, all_features


def analyze_feature_importance(X_train, y_train, X_val, y_val, feature_names):
    """分析特征重要性"""
    logger.info("=" * 60)
    logger.info("特征重要性分析 (SHAP)")
    logger.info("=" * 60)

    import lightgbm as lgb

    # 训练一个基础模型
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        random_state=42,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=31,
        n_estimators=500,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # 计算特征重要性
    importance = model.feature_importances_

    # 创建特征重要性 DataFrame
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    logger.info("\n特征重要性排名:")
    for i, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.0f}")

    # 保存特征重要性
    importance_df.to_csv("models/feature_importance.csv", index=False)

    return importance_df


def optimize_hyperparameters_optuna(X_train, y_train, X_val, y_val, n_trials=50):
    """使用 Optuna 优化超参数"""
    logger.info("=" * 60)
    logger.info(f"超参数优化 (Optuna, n_trials={n_trials})")
    logger.info("=" * 60)

    import optuna
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        }

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            verbosity=-1,
            random_state=42,
            n_estimators=500,
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        return -log_loss(y_val, val_pred)  # 最大化负 LogLoss

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value = -study.best_value

    logger.info(f"\n最佳参数:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"\n最佳 LogLoss: {best_value:.4f}")

    return best_params, best_value


def optimize_feature_selection(X_train, y_train, X_val, y_val, feature_names, importance_df):
    """基于特征重要性优化特征选择"""
    logger.info("=" * 60)
    logger.info("特征选择优化")
    logger.info("=" * 60)

    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score

    # 测试不同数量的特征
    results = []

    for n_features in range(10, len(feature_names) + 1, 2):
        # 选择前 N 个重要特征
        top_features = importance_df.head(n_features)["feature"].tolist()

        # 过滤存在的特征
        top_features = [f for f in top_features if f in X_train.columns]

        if len(top_features) < n_features:
            continue

        X_train_subset = X_train[top_features]
        X_val_subset = X_val[top_features]

        # 训练模型
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            verbosity=-1,
            random_state=42,
            max_depth=5,
            learning_rate=0.03,
            n_estimators=500,
        )

        model.fit(
            X_train_subset,
            y_train,
            eval_set=[(X_val_subset, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_pred = model.predict_proba(X_val_subset)[:, 1]
        ll = log_loss(y_val, val_pred)
        auc = roc_auc_score(y_val, val_pred)

        results.append(
            {"n_features": n_features, "log_loss": ll, "auc": auc, "features": top_features}
        )

        logger.info(f"特征数 {n_features}: LogLoss={ll:.4f}, AUC={auc:.4f}")

    # 找到最佳特征数
    best_result = min(results, key=lambda x: x["log_loss"])

    logger.info(f"\n最佳特征数: {best_result['n_features']}")
    logger.info(f"最佳 LogLoss: {best_result['log_loss']:.4f}")

    return best_result


def main() -> None:
    """主函数"""
    logger.info("")
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║   Happy-Badminton SOTA 优化            ║")
    logger.info("╚════════════════════════════════════════╝")
    logger.info("")

    config = load_config()

    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_optimization_data(
        config
    )

    # 1. 特征重要性分析
    importance_df = analyze_feature_importance(X_train, y_train, X_val, y_val, feature_names)

    # 2. 特征选择优化
    best_feature_result = optimize_feature_selection(
        X_train, y_train, X_val, y_val, feature_names, importance_df
    )

    # 3. 超参数优化（使用最佳特征子集）
    best_features = best_feature_result["features"]
    X_train_opt = X_train[best_features]
    X_val_opt = X_val[best_features]

    best_params, best_ll = optimize_hyperparameters_optuna(
        X_train_opt, y_train, X_val_opt, y_val, n_trials=30
    )

    # 4. 在测试集上评估最佳配置
    logger.info("=" * 60)
    logger.info("测试集最终评估")
    logger.info("=" * 60)

    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, accuracy_score

    X_test_opt = X_test[best_features]

    final_model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        random_state=42,
        n_estimators=500,
        **best_params,
    )

    final_model.fit(X_train_opt, y_train)

    test_pred = final_model.predict_proba(X_test_opt)[:, 1]

    test_ll = log_loss(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    test_bs = brier_score_loss(y_test, test_pred)
    test_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))

    logger.info("")
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║       优化后模型性能                    ║")
    logger.info("╠════════════════════════════════════════╣")
    logger.info(f"║  LogLoss:     {test_ll:.4f}                  ║")
    logger.info(f"║  AUC:         {test_auc:.4f}                  ║")
    logger.info(f"║  Brier Score: {test_bs:.4f}                  ║")
    logger.info(f"║  Accuracy:    {test_acc:.4f}                  ║")
    logger.info(f"║  特征数:      {len(best_features)}                    ║")
    logger.info("╚════════════════════════════════════════╝")
    logger.info("")

    # 保存结果
    results = {
        "best_params": best_params,
        "best_features": best_features,
        "test_metrics": {
            "log_loss": test_ll,
            "auc": test_auc,
            "brier_score": test_bs,
            "accuracy": test_acc,
        },
    }

    import json

    with open("models/optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("优化完成！结果已保存到 models/optimization_results.json")


if __name__ == "__main__":
    main()
