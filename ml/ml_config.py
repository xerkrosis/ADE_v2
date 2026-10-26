# Shared settings for all ML scripts

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "rf"
SPLIT_DIR = RESULTS_DIR / "splits"
MODEL_DIR = RESULTS_DIR / "models"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
PREDICTION_DIR = RESULTS_DIR / "predictions"

# Keep generated model paths short enough for the Visual Studio Python
# file watcher on Windows. The complete experiment configuration remains
# stored inside every joblib bundle.
MODEL_NAMING_VERSION = "compact_v1"
MAX_MODEL_PATH_LENGTH = 240

# Data used as model input
FEATURE_VARIANT = "original_v2"
FEATURE_DATA_DIR = Path(os.getenv(
    "ML_FEATURE_DATA_DIR",
    str(PROJECT_ROOT / "results" / "Artifact_annotations_pipeline" / "original_w5_o0_fixedthr"),
))

TRAIN_DATA_DIR = FEATURE_DATA_DIR
TEST_DATA_DIR = FEATURE_DATA_DIR

# Fixed labels always come from original_v2
LABEL_VARIANT = "original_v2"
LABEL_DATA_DIR = Path(os.getenv("ML_LABEL_DATA_DIR", str(FEATURE_DATA_DIR)))

# Create/use your own splits from current original exports
SPLIT_VARIANT = "original_v2"

# Only distinguish within originally usable windows
COHORT_FILTER = "usable_only"

# Choose Quality Labels: "QUALITY_BASE_LABEL", "QUALITY_LABEL", "USABLE_LABEL", "STRICT_CLEAN_LABEL"
TARGET_COLUMNS = [
    # "USABLE_LABEL",
    "STRICT_CLEAN_LABEL",
]

TRAINING_VARIANT = (
    f"{FEATURE_VARIANT}"
    f"_labels-{LABEL_VARIANT}"
    f"_cohort-{COHORT_FILTER}"
)
TEST_VARIANT = "original_v2"

# "files": complete recordings stay together
# "windows": individual windows are distributed independently
SPLIT_MODE = os.getenv("ML_SPLIT_MODE", "files")  # "windows", "files"
SPLIT_VERSION = "v1"        # "v1", "v2", ...

# WARNING: Verify that TRAIN_SIZE, VALIDATION_SIZE and TEST_SIZE add up to 1.00!
TRAIN_SIZE = float(os.getenv("ML_TRAIN_SIZE", "0.70"))           # Custom: 0.60 ... 0.75
TEST_SIZE = 0.0
VALIDATION_SIZE = 1.0 - TRAIN_SIZE - TEST_SIZE
SPLIT_RANDOM_STATE = int(os.getenv("ML_SPLIT_RANDOM_STATE", "42")) # 11, 23, 42, 67, 89, or other seed-number
MODEL_RANDOM_STATE = 42 # Default: 42
RANDOM_STATE = 42       # Legacy-Seed

SPLIT_ID = (
    f"{SPLIT_MODE}_"
    f"train{TRAIN_SIZE * 100:g}_"
    f"validation{VALIDATION_SIZE * 100:g}_"
    f"test{TEST_SIZE * 100:g}_"
    f"{SPLIT_VERSION}_"
    f"splitseed{SPLIT_RANDOM_STATE}"
).replace(".", "p")
SPLIT_FILE = SPLIT_DIR / (
    f"split_{SPLIT_VARIANT}_{SPLIT_ID}.csv"
)
SPLIT_DATA_DIR = LABEL_DATA_DIR

# Console Output for visual confirmation
def print_configuration():
    """Print the currently selected ML configuration."""
    print("\nSelected ML configuration:")
    print(f"  Split mode: {SPLIT_MODE}")
    print(f"  Split version: {SPLIT_VERSION}")
    print(
        "  Train / validation / test: "
        f"{TRAIN_SIZE * 100:g}% / "
        f"{VALIDATION_SIZE * 100:g}% / "
        f"{TEST_SIZE * 100:g}%"
    )
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  Target columns: {', '.join(TARGET_COLUMNS)}")
    print()

if __name__ == "__main__":
    print_configuration()
