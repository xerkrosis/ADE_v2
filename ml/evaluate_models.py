"""Evaluate locally trained RF bundles on their saved held-out partition."""
import argparse
import hashlib
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, recall_score,
)

from .ml_config import EVALUATION_DIR, MODEL_DIR, SPLIT_DIR, TRAINING_VARIANT
from .ml_utils import load_feature_data_with_fixed_labels, select_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", help="Explicit model path; repeatable")
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    # Compact model names are selected by bundle metadata, not legacy filenames.
    paths = args.model or sorted(MODEL_DIR.glob("*.joblib"))
    bundles = []
    for path in paths:
        bundle = joblib.load(path)
        if (bundle.get("cohort_filter") != "usable_only"
                or bundle.get("target") != "STRICT_CLEAN_LABEL"):
            if args.model:
                raise ValueError(f"Incompatible target/cohort: {path}")
            continue
        if not args.model and bundle.get("training_variant") != TRAINING_VARIANT:
            continue
        bundles.append((path, bundle))
    if not bundles:
        raise FileNotFoundError(f"No matching RF bundles in {MODEL_DIR}; train a model first.")
    data, _ = load_feature_data_with_fixed_labels(show_progress=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, bundle in bundles:
        split_path = SPLIT_DIR / Path(bundle["split_file"]).name
        if not split_path.is_file():
            raise FileNotFoundError(f"This model's own split is required: {split_path}")
        evaluation_data = select_partition(data, split_path, bundle["split_mode"], args.partition)
        if len(evaluation_data) == 0:
            raise ValueError(f"Empty {args.partition} partition: {split_path}")
        actual = evaluation_data[bundle["target"]].to_numpy()
        if set(np.unique(actual)) != {0, 1}:
            raise ValueError("Evaluation requires both classes for sensitivity/specificity.")
        prediction = bundle["model"].predict(evaluation_data[bundle["feature_columns"]])
        matrix = confusion_matrix(actual, prediction, labels=[0, 1])
        normalized_matrix = confusion_matrix(actual, prediction, labels=[0, 1], normalize="true")
        row = {
            "model_file": path.name, "target": bundle["target"],
            "split_mode": bundle["split_mode"], "partition": args.partition,
            "split_seed": bundle.get("split_random_state"),
            "n": len(actual), "f1_score": f1_score(actual, prediction),
            "accuracy": accuracy_score(actual, prediction),
            "balanced_accuracy": balanced_accuracy_score(actual, prediction),
            "sensitivity": recall_score(actual, prediction, pos_label=1),
            "specificity": recall_score(actual, prediction, pos_label=0),
        }
        rows.append(row)
        # All metadata stays in metrics.csv; filesystem paths stay short.
        tag = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
        for values, suffix in ((matrix, "counts"), (normalized_matrix, "normalized")):
            pd.DataFrame(values, index=["actual_0", "actual_1"],
                         columns=["predicted_0", "predicted_1"]).to_csv(
                EVALUATION_DIR / f"cm_{tag}_{suffix}.csv"
            )
        figure, axis = plt.subplots(figsize=(6, 5))
        try:
            ConfusionMatrixDisplay(normalized_matrix,
                                   display_labels=["usable_not_strict", "strict_clean"]).plot(
                ax=axis, cmap="Blues", values_format=".1%", colorbar=False
            )
            axis.set_title(f"RF — {bundle['split_mode']} — {args.partition}")
            figure.savefig(EVALUATION_DIR / f"cm_{tag}.png", dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(EVALUATION_DIR / f"metrics_{args.partition}.csv", index=False)
    print("Evaluation output:", EVALUATION_DIR)


if __name__ == "__main__":
    main()
