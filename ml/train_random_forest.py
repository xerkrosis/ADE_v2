# --- Train ---
# Random Forest

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .ml_config import MODEL_RANDOM_STATE, TARGET_COLUMNS
from .ml_utils import load_training_data, save_model

try:
    import winsound
except ImportError:
    winsound = None


N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "300"))
CRITERION = "gini"
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

CLASS_WEIGHT_NAME = os.getenv("ML_CLASS_WEIGHT", "balanced").strip().lower()
CLASS_WEIGHT_OPTIONS = {
    "none": None,
    "balanced": "balanced",
}

if CLASS_WEIGHT_NAME not in CLASS_WEIGHT_OPTIONS:
    raise ValueError(
        "ML_CLASS_WEIGHT must be 'none' or 'balanced', got "
        f"{CLASS_WEIGHT_NAME!r}."
    )

CLASS_WEIGHT = CLASS_WEIGHT_OPTIONS[CLASS_WEIGHT_NAME]

MODEL_TAG = (
    f"random-forest_n-{N_ESTIMATORS}_{CRITERION}_"
    f"depth-{MAX_DEPTH}_features-{MAX_FEATURES}_"
    f"classweight-{CLASS_WEIGHT_NAME}"
)


def main():
    train_data, features = load_training_data()

    for target in TARGET_COLUMNS:
        target_classes = set(
            train_data[target]
            .dropna()
            .unique()
            .tolist()
        )

        if target_classes != {0, 1}:
            raise ValueError(
                f"Target {target} must contain both binary classes 0 and 1; "
                f"got {sorted(target_classes)}."
            )

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                criterion=CRITERION,
                max_depth=MAX_DEPTH,
                max_features=MAX_FEATURES,
                class_weight=CLASS_WEIGHT,
                random_state=MODEL_RANDOM_STATE,
                n_jobs=-1,
            ),
        )
        print(
            f"Fitting RF{N_ESTIMATORS} for {target} on "
            f"{len(train_data):,} windows with {len(features)} features "
            f"(class_weight={CLASS_WEIGHT_NAME}) ...",
            flush=True,
        )
        model.fit(train_data[features], train_data[target])
        output_path = save_model(model, MODEL_TAG, target, features)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Training: Random Forest")
    print(f"n_estimators: {N_ESTIMATORS}")
    print(f"class_weight: {CLASS_WEIGHT_NAME}")
    main()
    if winsound is not None:
        winsound.MessageBeep(winsound.MB_ICONASTERISK)