# Create shared training-split

import pandas as pd
from sklearn.model_selection import train_test_split

from .ml_config import (
    SPLIT_DATA_DIR,
    SPLIT_RANDOM_STATE,
    SPLIT_FILE,
    SPLIT_MODE,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)
from .ml_utils import load_data

try:
    import winsound
except ImportError:
    winsound = None
import os

def main():
    if round(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE, 10) != 1.0:
        raise ValueError("TRAIN_SIZE, VALIDATION_SIZE and TEST_SIZE must sum to 1.")

    if SPLIT_FILE.exists():
        if os.getenv("ML_REUSE_EXISTING_SPLIT") == "1":
            print(f"Reusing existing split: {SPLIT_FILE}")
            return

        raise FileExistsError(
            f"Split already exists: {SPLIT_FILE}\n"
            "Change SPLIT_VERSION or SPLIT_RANDOM_STATE for a new split."
        )

    print(f"Loading data from: {SPLIT_DATA_DIR}")
    data = load_data(SPLIT_DATA_DIR, show_progress=True)

    print(
        f"Loaded {data['recording_id'].nunique()} recordings "
        f"with {len(data)} windows."
    )
    print("Creating split...")

    if SPLIT_MODE == "files":
        items = pd.DataFrame(
            {"recording_id": sorted(data["recording_id"].unique())}
        )
    elif SPLIT_MODE == "windows":
        items = data[["recording_id", "window_number"]].copy()
        print("Warning: overlapping windows may enter different partitions.")
    else:
        raise ValueError("SPLIT_MODE must be 'files' or 'windows'.")

    remaining_size = VALIDATION_SIZE + TEST_SIZE

    train_items, remaining_items = train_test_split(
        items,
        train_size=TRAIN_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )

    if TEST_SIZE == 0:
        validation_items = remaining_items
        test_items = remaining_items.iloc[0:0].copy()
    elif VALIDATION_SIZE == 0:
        validation_items = remaining_items.iloc[0:0].copy()
        test_items = remaining_items
    else:
        validation_items, test_items = train_test_split(
            remaining_items,
            train_size=VALIDATION_SIZE / remaining_size,
            random_state=SPLIT_RANDOM_STATE,
        )

    split = pd.concat(
        [
            train_items.assign(split="train"),
            validation_items.assign(split="validation"),
            test_items.assign(split="test"),
        ],
        ignore_index=True,
    )

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT_FILE, index=False)

    print(split["split"].value_counts())
    print(f"Saved: {SPLIT_FILE}")

if __name__ == "__main__":
    main()
    if winsound is not None:
        winsound.MessageBeep(winsound.MB_ICONASTERISK)  # Done SFX