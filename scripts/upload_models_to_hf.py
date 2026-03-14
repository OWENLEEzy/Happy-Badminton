"""Upload trained models to HuggingFace Model Hub.

Run this script once to upload models:
    uv run python scripts/upload_models_to_hf.py

Then the HuggingFace Space will download them automatically on startup.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from huggingface_hub import HfApi, login
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install huggingface_hub first:")
    print("  uv add huggingface_hub")
    print("\nThen run this script again.")
    sys.exit(1)


def upload_models():
    """Upload models to HuggingFace Model Hub."""
    print("=" * 60)
    print("Uploading models to HuggingFace Model Hub")
    print("=" * 60)

    # Login to HuggingFace
    print("\n1. Logging in to HuggingFace...")
    try:
        # Try to get token from environment variable
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
            print("   ✓ Logged in successfully (using HF_TOKEN env var)")
        else:
            # Fallback to interactive login
            login()
            print("   ✓ Logged in successfully (interactive)")
    except Exception as e:
        print(f"   ✗ Login failed: {e}")
        print("\nPlease set HF_TOKEN environment variable or run: huggingface-cli login")
        return

    api = HfApi()
    repo_id = "owenlee-5678/happy-badminton-models"

    # Create repository if it doesn't exist
    print(f"\n2. Creating repository: {repo_id}")
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        print(f"   ✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"   ⚠ Repository creation warning: {e}")

    # Upload model files
    models_dir = project_root / "models"

    files_to_upload = [
        # Main prediction model
        "simplified_ensemble.pkl",
        "simplified_results.json",
        "simplified_feature_importance.json",
        "nat_pair_win_rates.json",
        # Set count prediction model
        "set_count_model.pkl",
        "set_count_results.json",
    ]

    print("\n3. Uploading model files...")
    for filename in files_to_upload:
        filepath = models_dir / filename
        if not filepath.exists():
            print(f"   ⚠ Skipping {filename} (not found)")
            continue

        print(f"   ↑ Uploading {filename}...")
        try:
            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_type="model",
            )
            print(f"   ✓ Uploaded {filename}")
        except Exception as e:
            print(f"   ✗ Failed to upload {filename}: {e}")

    # Create model card
    print("\n4. Creating model card...")
    model_card = """---
license: mit
tags:
  - badminton
  - sports-prediction
  - ensemble-model
  - lightgbm
  - xgboost
  - catboost
---

# Happy Badminton Prediction Models

Trained machine learning models for badminton match prediction.

## Models

### Simplified Ensemble (simplified_ensemble.pkl)
- **Task**: Binary classification (predict match winner)
- **Framework**: Custom stacking ensemble (LightGBM + XGBoost + CatBoost → BayesianRidge meta-learner)
- **Features**: 35 pre-match features (ranking, form, streak, H2H, nationality)
- **Performance**:
  - AUC: 0.9608
  - LogLoss: 0.2316
  - Brier Score: 0.0722

### Set Count Model (set_count_model.pkl)
- **Task**: Binary classification (predict if match goes to 2 or 3 sets)
- **Framework**: StackingEnsemble (same architecture)
- **Features**: 31 pre-match features + historical 3-set rates
- **Performance**:
  - AUC: 0.6635
  - LogLoss: 0.5583

## Usage

```python
from huggingface_hub import hf_hub_download
import joblib

# Download main model
model_path = hf_hub_download(
    repo_id="owenlee-5678/happy-badminton-models",
    filename="simplified_ensemble.pkl"
)
model = joblib.load(model_path)

# Download set count model
set_count_path = hf_hub_download(
    repo_id="owenlee-5678/happy-badminton-models",
    filename="set_count_model.pkl"
)
set_count_model = joblib.load(set_count_path)
```

## Training Data

- **Source**: BWF official tournament records (2019-2025)
- **Matches**: ~15,000 professional matches
- **Split**: Time-based (70% train, 15% val, 15% test)

## Feature Schema

See `simplified_results.json` for the complete feature list.

## Citation

```bibtex
@software{happy_badminton_2026,
  title={Happy Badminton Prediction Models},
  author={OWENLEE},
  year={2026},
  url={https://huggingface.co/owenlee-5678/happy-badminton-models}
}
```
"""

    try:
        # Use /tmp directory to avoid deleting project README.md
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            readme_path = Path(f.name)
            f.write(model_card)

        try:
            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_type="model",
            )
            print("   ✓ Model card created")
        finally:
            # Clean up temporary README (safe to delete from /tmp)
            readme_path.unlink()
    except Exception as e:
        print(f"   ✗ Failed to create model card: {e}")

    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)
    print(f"\nModel Hub: https://huggingface.co/{repo_id}")
    print("\nNext steps:")
    print("1. Verify models are available on the Model Hub page")
    print("2. Deploy the HuggingFace Space (it will auto-download models)")


if __name__ == "__main__":
    upload_models()
