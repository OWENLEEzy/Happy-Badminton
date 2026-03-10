#!/usr/bin/env python3
"""
数据可视化生成脚本

加载原始数据并生成 docs/plots/ 下的全部分析图表。

运行:
    uv run python scripts/generate_plots.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.utils.logger import setup_logger
from src.data.loader import load_and_merge
from src.data.preprocessor import preprocess_pipeline
from src.visualization.plots import create_all_visualizations

logger = setup_logger()


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    logger.info("Loading data...")
    df = load_and_merge(config["data"]["raw_path"])
    df = preprocess_pipeline(df)
    logger.info(f"Dataset ready: {df.shape[0]:,} matches × {df.shape[1]} columns")

    create_all_visualizations(df, output_dir="docs/plots")


if __name__ == "__main__":
    main()
