#!/usr/bin/env python3
"""
数据准备脚本

执行完整的数据加载和预处理流程
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_and_merge
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import setup_logger


def main() -> None:
    """主函数"""
    # 设置日志
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("开始数据准备流程")
    logger.info("=" * 60)

    # 配置
    INPUT_PATH = "data/raw/Tournament Results.xlsx"
    OUTPUT_PATH = "data/processed/matches_clean.parquet"

    # 步骤 1: 加载数据
    logger.info("\n步骤 1: 加载 Excel 数据")
    df = load_and_merge(INPUT_PATH)

    # 步骤 2: 预处理
    logger.info("\n步骤 2: 数据预处理")
    preprocessor = DataPreprocessor(df)

    preprocessor.filter_future_dates().identify_and_filter_retirements().handle_duration_outliers().parse_scores().handle_missing_values().sort_by_date().add_target_variable()

    # 步骤 3: 保存
    logger.info("\n步骤 3: 保存处理后的数据")
    preprocessor.save_processed(OUTPUT_PATH)

    # 摘要
    logger.info(f"\n{'=' * 60}")
    logger.info("数据准备完成！")
    logger.info("=" * 60)
    summary = preprocessor.get_summary()
    logger.info(f"原始数据: {summary['original_shape']}")
    logger.info(f"最终数据: {summary['final_shape']}")

    # 打印统计
    stats = summary["stats"]
    logger.info(f"\n过滤统计:")
    logger.info(f"  未来日期: {stats.get('future_dates_filtered', 0)} 条")
    logger.info(
        f"  退赛比赛: {stats.get('retirement_matches', 0)} 场 ({stats.get('retirement_rate', 0):.2f}%)"
    )
    logger.info(f"  0分钟时长: {stats.get('zero_duration', 0)} 场")
    logger.info(f"  极值时长: {stats.get('extreme_duration', 0)} 场")

    # 数据预览
    logger.info(f"\n最终数据预览:")
    logger.info(
        f"  日期范围: {preprocessor.df['match_date'].min()} ~ {preprocessor.df['match_date'].max()}"
    )
    logger.info(
        f"  比赛类型: MS={preprocessor.df['type'].eq('MS').sum()}, WS={preprocessor.df['type'].eq('WS').sum()}"
    )
    logger.info(
        f"  局数分布: 2局={(preprocessor.df['sets_played'] == 2).sum()}, 3局={(preprocessor.df['sets_played'] == 3).sum()}"
    )

    return preprocessor.df


if __name__ == "__main__":
    main()
