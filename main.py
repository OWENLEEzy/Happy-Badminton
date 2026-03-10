"""Project entry point.

Usage:
    uv run python main.py          # auto-train if needed, then start server
    uv run python main.py --train  # force re-train even if model exists
    uv run python main.py --port 8080
"""

import argparse
import os
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Tournament Results.xlsx"
MODEL_FILE = PROJECT_ROOT / "models" / "simplified_ensemble.pkl"
RESULTS_FILE = PROJECT_ROOT / "models" / "simplified_results.json"
SET_COUNT_MODEL = PROJECT_ROOT / "models" / "set_count_model.pkl"
SET_COUNT_RESULTS = PROJECT_ROOT / "models" / "set_count_results.json"

sys.path.insert(0, str(PROJECT_ROOT))


def _banner(msg: str) -> None:
    width = 60
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def check_data() -> None:
    """Fail fast with a friendly message if raw data is missing."""
    if not DATA_FILE.exists():
        _banner("ERROR: Data file not found")
        print(f"  Expected: {DATA_FILE}")
        print()
        print("  Place 'Tournament Results.xlsx' in data/raw/ and retry.")
        print("  (Sample data: data/raw/Tournament Results - Sample.xlsx)")
        sys.exit(1)


def model_ready() -> bool:
    """Return True only when all model artefacts exist."""
    return (
        MODEL_FILE.exists()
        and RESULTS_FILE.exists()
        and SET_COUNT_MODEL.exists()
        and SET_COUNT_RESULTS.exists()
    )


def run_training() -> None:
    """Run training scripts in-process (preserves live log output)."""
    _banner("Training models (first run ~5-8 min)")
    print()
    os.chdir(PROJECT_ROOT)
    runpy.run_path(
        str(PROJECT_ROOT / "scripts" / "train_simplified.py"),
        run_name="__main__",
    )
    runpy.run_path(
        str(PROJECT_ROOT / "scripts" / "train_set_count.py"),
        run_name="__main__",
    )


def start_server(port: int) -> None:
    """Import and run the Flask app."""
    from frontend.app import app  # noqa: PLC0415

    print()
    _banner("Badminton Prediction — General Predictor")
    print(f"  Open: http://localhost:{port}")
    print("=" * 60)
    print()
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Happy-Badminton entry point")
    parser.add_argument("--train", action="store_true", help="Force re-train the model")
    parser.add_argument("--port", type=int, default=5001, help="Server port (default: 5001)")
    args = parser.parse_args()

    check_data()

    if args.train or not model_ready():
        run_training()

    start_server(args.port)


if __name__ == "__main__":
    main()
