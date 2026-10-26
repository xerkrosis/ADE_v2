# Shared file loading and split access

import hashlib
import re
from pathlib import Path

import joblib
import pandas as pd

from .ml_config import (
    COHORT_FILTER,
    FEATURE_DATA_DIR,
    FEATURE_VARIANT,
    LABEL_DATA_DIR,
    LABEL_VARIANT,
    MAX_MODEL_PATH_LENGTH,
    MODEL_DIR,
    MODEL_NAMING_VERSION,
    MODEL_RANDOM_STATE,
    SPLIT_FILE,
    SPLIT_MODE,
    SPLIT_RANDOM_STATE,
    TRAIN_DATA_DIR,
    TRAINING_VARIANT,
)

def find_files(path):
    path = Path(path)

    if path.is_file():
        return [path]

    files = sorted(
        file_path
        for file_path in path.glob("*_artifacts.xlsx")
        if not file_path.name.startswith("~$")
    )

    if not files:
        raise FileNotFoundError(f"No pipeline exports found in: {path}")

    return files

def load_data(path, show_progress=False):
    frames = []
    files = find_files(path)
    total_files = len(files)

    for file_number, file_path in enumerate(files, start=1):
        if show_progress:
            print(f"Loading {file_number}/{total_files}: {file_path.name} ...", flush=True)
        frame = pd.read_excel(
            file_path,
            sheet_name="windows",
            engine="openpyxl",
            index_col=0,
        )
        metadata = pd.DataFrame(
            {
                "recording_id": file_path.name.removesuffix("_artifacts.xlsx"),
                "window_number": frame.index,
            },
            index=frame.index,
        )

        frame = pd.concat([metadata, frame], axis=1)
        frames.append(frame.reset_index(drop=True))
        
        if show_progress:
            print(
                f"\rLoading files: {file_number}/{total_files} "
                f"({file_number / total_files:.0%})",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    return pd.concat(frames, ignore_index=True)

def get_feature_columns(data):
    """Pipeline features are numeric/lower-case; labels are upper-case."""
    metadata = {"begin_idx", "end_idx", "window_number"}

    return [
        column
        for column in data.select_dtypes(include="number").columns
        if column.islower() and column not in metadata
    ]

WINDOW_KEY_COLUMNS = [
    "recording_id",
    "window_number",
]

FIXED_LABEL_COLUMNS = [
    "USABLE_LABEL",
    "STRICT_CLEAN_LABEL",
]


def load_feature_data_with_fixed_labels(
    feature_path=FEATURE_DATA_DIR,
    label_path=LABEL_DATA_DIR,
    cohort_filter=COHORT_FILTER,
    show_progress=False,
):
    """
    Load numeric features from one pipeline variant while keeping
    USABLE_LABEL and STRICT_CLEAN_LABEL fixed to another variant.
    """

    print(f"Loading features from: {feature_path}")

    feature_data = load_data(
        feature_path,
        show_progress=show_progress,
    )

    feature_columns = get_feature_columns(feature_data)

    same_source = (
        Path(feature_path).resolve()
        == Path(label_path).resolve()
    )

    if same_source:
        label_data = feature_data
    else:
        print(f"Loading fixed labels from: {label_path}")

        label_data = load_data(
            label_path,
            show_progress=show_progress,
        )

    required_label_columns = (
        WINDOW_KEY_COLUMNS
        + ["begin_idx", "end_idx"]
        + FIXED_LABEL_COLUMNS
    )

    missing_label_columns = [
        column
        for column in required_label_columns
        if column not in label_data.columns
    ]

    if missing_label_columns:
        raise KeyError(
            "Missing columns in fixed-label data: "
            + ", ".join(missing_label_columns)
        )

    for data_name, data in [
        ("feature data", feature_data),
        ("label data", label_data),
    ]:
        duplicate_mask = data.duplicated(
            WINDOW_KEY_COLUMNS,
            keep=False,
        )

        if duplicate_mask.any():
            duplicate_count = int(duplicate_mask.sum())

            raise ValueError(
                f"{data_name} contains {duplicate_count} "
                "duplicate recording/window keys."
            )

    fixed_labels = (
        label_data[required_label_columns]
        .rename(
            columns={
                "begin_idx": "_label_begin_idx",
                "end_idx": "_label_end_idx",
            }
        )
        .copy()
    )

    # Remove method-specific target columns from the feature dataset.
    # They will be replaced by the fixed original_v2 labels.
    feature_data = feature_data.drop(
        columns=FIXED_LABEL_COLUMNS,
        errors="ignore",
    ).copy()

    merged = feature_data.merge(
        fixed_labels,
        on=WINDOW_KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_counts = (
        merged["_merge"]
        .value_counts()
        .to_dict()
    )

    if unmatched_counts.get("left_only", 0) > 0:
        raise ValueError(
            "Feature windows without matching original labels: "
            f"{unmatched_counts['left_only']}"
        )

    if unmatched_counts.get("right_only", 0) > 0:
        raise ValueError(
            "Original-label windows without matching features: "
            f"{unmatched_counts['right_only']}"
        )

    merged = merged.drop(columns="_merge")

    begin_matches = (
        merged["begin_idx"]
        == merged["_label_begin_idx"]
    )

    end_matches = (
        merged["end_idx"]
        == merged["_label_end_idx"]
    )

    if not begin_matches.all() or not end_matches.all():
        mismatch_count = int(
            (~(begin_matches & end_matches)).sum()
        )

        raise ValueError(
            "Window boundaries differ between feature and label "
            f"datasets for {mismatch_count} windows."
        )

    merged = merged.drop(
        columns=[
            "_label_begin_idx",
            "_label_end_idx",
        ]
    )

    invalid_label_order = (
        (merged["STRICT_CLEAN_LABEL"] == 1)
        & (merged["USABLE_LABEL"] != 1)
    )

    if invalid_label_order.any():
        raise ValueError(
            "Found STRICT_CLEAN_LABEL=1 outside USABLE_LABEL=1: "
            f"{int(invalid_label_order.sum())} windows."
        )

    total_windows = len(merged)

    if cohort_filter == "usable_only":
        merged = merged[
            merged["USABLE_LABEL"] == 1
        ].copy()

    elif cohort_filter not in (None, "all"):
        raise ValueError(
            "COHORT_FILTER must be 'usable_only', 'all' or None."
        )

    print(
        f"Matched windows: {total_windows}"
        f"\nSelected cohort: {len(merged)}"
        f"\nFeature count: {len(feature_columns)}"
    )

    print(
        "\nTarget distribution:"
        f"\n{merged['STRICT_CLEAN_LABEL'].value_counts().sort_index()}"
    )

    return merged, feature_columns

def select_partition(data, split_file, split_mode, partition):
    assignments = pd.read_csv(
        split_file,
        dtype={"recording_id": str},
    )
    keys = ["recording_id"]

    if split_mode == "windows":
        keys.append("window_number")

    data_with_split = data.merge(assignments, on=keys, how="inner")
    return data_with_split[data_with_split["split"] == partition].copy()


def load_training_data():
    data, feature_columns = (
        load_feature_data_with_fixed_labels(
            show_progress=True,
        )
    )

    train_data = select_partition(
        data,
        SPLIT_FILE,
        SPLIT_MODE,
        "train",
    )

    print(
        "\nTraining target distribution:"
        f"\n{train_data['STRICT_CLEAN_LABEL'].value_counts().sort_index()}"
    )

    return train_data, feature_columns

def _filename_token(value):
    """Return a compact, filesystem-safe lowercase token."""
    token = str(value).strip().lower()
    replacements = (
        ("random-forest_n-", "rf"),
        ("dummy-classifier_", "dummy-"),
        ("decision-tree_", "dt-"),
        ("_depth-", "-d-"),
        ("_features-", "-f-"),
        ("_classweight-", "-cw-"),
    )
    for old, new in replacements:
        token = token.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "-", token).strip("-")


def build_model_filename(
    model_tag,
    target,
    *,
    split_file_name,
    split_mode,
    split_random_state,
    model_random_state,
    training_variant,
):
    """Build a short deterministic name while retaining full metadata."""
    compact_tag = _filename_token(model_tag)
    target_token = _filename_token(
        str(target).removesuffix("_LABEL")
    )
    split_token = _filename_token(split_mode)

    signature = "\n".join(
        [
            MODEL_NAMING_VERSION,
            compact_tag,
            str(target),
            str(training_variant),
            str(split_file_name),
            str(split_mode),
            str(split_random_state),
            str(model_random_state),
        ]
    )
    experiment_id = hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()[:10]

    return (
        f"{compact_tag}__{target_token}__"
        f"{split_token}-s{split_random_state}-m{model_random_state}__"
        f"{experiment_id}.joblib"
    )


def ensure_model_path_is_safe(path):
    """Reject model paths likely to crash legacy Windows file watchers."""
    path = Path(path)
    path_length = len(str(path.resolve()))
    if path_length > MAX_MODEL_PATH_LENGTH:
        raise ValueError(
            f"Model path has {path_length} characters; the configured "
            f"limit is {MAX_MODEL_PATH_LENGTH}: {path}"
        )
    return path


def model_path(model_tag, target):
    filename = build_model_filename(
        model_tag,
        target,
        split_file_name=SPLIT_FILE.name,
        split_mode=SPLIT_MODE,
        split_random_state=SPLIT_RANDOM_STATE,
        model_random_state=MODEL_RANDOM_STATE,
        training_variant=TRAINING_VARIANT,
    )
    return ensure_model_path_is_safe(MODEL_DIR / filename)

def save_model(model, model_tag, target, feature_columns):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = model_path(model_tag, target)

    joblib.dump(
        {
            "model": model,
            "model_tag": model_tag,
            "target": target,
            "feature_columns": feature_columns,
            "split_file": SPLIT_FILE.name,
            "split_mode": SPLIT_MODE,
            "split_random_state": SPLIT_RANDOM_STATE,
            "model_random_state": MODEL_RANDOM_STATE,
            "training_variant": TRAINING_VARIANT,
            "model_file": output_path.name,
            "model_naming_version": MODEL_NAMING_VERSION,

            # New experiment metadata
            "feature_variant": FEATURE_VARIANT,
            "feature_data_dir": str(FEATURE_DATA_DIR),
            "label_variant": LABEL_VARIANT,
            "label_data_dir": str(LABEL_DATA_DIR),
            "cohort_filter": COHORT_FILTER,

            # Raw Random-Forest predict_proba output. Set this to True only
            # for a model that was fitted with an explicit calibration step.
            "probability_calibrated": False,
        },
        output_path,
    )

    return output_path
