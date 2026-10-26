# Rule-based quality labels and optional probabilistic quality assessment

from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np

from .artifacts import pooled_quantile


ML_QUALITY_PROBABILITY_COLUMNS = [
    "ML_P_USABLE_NOT_STRICT",
    "ML_P_STRICT_CLEAN",
    "ML_STRICT_CLEAN_AT_THRESHOLD",
]


# Documented RF evaluation setup. Existing bundles record feature_variant,
# but do not store window length, overlap or sample rate. These reference
# values come from the original_v2 experiment, not from inferred metadata.
ML_QUALITY_REFERENCE_CONTEXTS = {
    "original_v2": {
        "signal_method": "original",
        "window_sec": 5.0,
        "window_overlap": 0.0,
        "sample_rate": 125.0,
    },
}


# OPTIONAL V2 EXTENSION:
def empty_ml_quality_summary(strict_clean_threshold):
    # Return the default per-recording ML summary
    return {
        "requested": False,
        "applied": False,
        "model_file": "",
        "model_tag": "",
        "training_variant": "",
        "feature_variant": "",
        "cohort_filter": "",
        "probability_calibrated": False,
        "strict_clean_threshold": float(strict_clean_threshold),
        "scored_windows": 0,
        "mean_p_usable_not_strict_all_usable": np.nan,
        "mean_p_strict_clean_all_usable": np.nan,
        "mean_p_usable_not_strict_rule_usable_not_strict": np.nan,
        "mean_p_strict_clean_rule_usable_not_strict": np.nan,
        "mean_p_usable_not_strict_rule_strict_clean": np.nan,
        "mean_p_strict_clean_rule_strict_clean": np.nan,
        "strict_clean_windows_at_threshold": 0,
        "strict_clean_share_at_threshold": np.nan,
        "application_signal_method": "",
        "application_window_sec": np.nan,
        "application_window_overlap": np.nan,
        "application_sample_rate": np.nan,
        "application_reference": "",
        "application_context_status": "not_applied",
        "application_context_warning": "",
    }


def describe_ml_application_context(
    model_bundle,
    signal_method,
    window_sec=None,
    window_overlap=None,
    sample_rate=None,
):
    """Describe transfer beyond the documented experiment without blocking it.

    Matching these fields is a context check, not proof of generalization.
    Other preprocessing settings and feature definitions must still match.
    """
    actual = {"signal_method": str(signal_method)}
    for key, value in (
        ("window_sec", window_sec),
        ("window_overlap", window_overlap),
        ("sample_rate", sample_rate),
    ):
        try:
            actual[key] = float(value)
        except (TypeError, ValueError):
            actual[key] = np.nan

    context = {
        "application_signal_method": actual["signal_method"],
        "application_window_sec": actual["window_sec"],
        "application_window_overlap": actual["window_overlap"],
        "application_sample_rate": actual["sample_rate"],
        "application_reference": "",
        "application_context_status": "reference_unknown",
        "application_context_warning": "",
    }
    feature_variant = model_bundle.get("feature_variant", "")
    reference = ML_QUALITY_REFERENCE_CONTEXTS.get(feature_variant)
    if reference is None:
        context["application_context_warning"] = (
            f"ML quality model feature_variant={feature_variant!r}: no "
            "documented signal/window reference is registered. Scores are "
            "provided, but application reliability and the chosen threshold "
            "have not been verified for this setup."
        )
        return context

    context["application_reference"] = (
        f"{feature_variant}: signal_method={reference['signal_method']}; "
        f"window_sec={reference['window_sec']:g}; "
        f"window_overlap={reference['window_overlap']:g}; "
        f"sample_rate={reference['sample_rate']:g} Hz"
    )
    differences = []
    if actual["signal_method"] != reference["signal_method"]:
        differences.append(f"signal_method={actual['signal_method']!r}")
    for key in ("window_sec", "window_overlap", "sample_rate"):
        if not np.isfinite(actual[key]):
            differences.append(f"{key}=unknown")
        elif not np.isclose(actual[key], reference[key], rtol=0.0, atol=1e-12):
            differences.append(f"{key}={actual[key]:g}")

    context["application_context_status"] = (
        "outside_reference" if differences else "matches_reference_fields"
    )
    if differences:
        context["application_context_warning"] = (
            "ML quality scores are being applied outside the documented "
            f"reference setup ({context['application_reference']}). "
            f"Current differences: {', '.join(differences)}. "
            "Scores are provided, but reliability and the chosen threshold "
            "have not been validated for this changed setup."
        )
    return context


@lru_cache(maxsize=None)
def load_ml_quality_model_bundle(model_path):
    # Load and cache one model bundle for repeated batch runs
    try:
        import joblib
    except ImportError as error:
        raise ImportError(
            "joblib is required when apply_ml_quality_model=True."
        ) from error

    resolved_path = Path(model_path).expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"ML quality model not found: {resolved_path}"
        )

    bundle = joblib.load(resolved_path)

    if not isinstance(bundle, dict):
        raise TypeError(
            "The ML quality model file must contain a model bundle dictionary."
        )

    return bundle


def add_ml_quality_probabilities(
    df_features,
    model_bundle,
    strict_clean_threshold,
):
    """
    Add probabilities within the rule-based usable cohort.

    Class 0 is usable-but-not-strict; class 1 is strict-clean. The original
    rule-based labels are preserved. Non-usable windows remain unscored.
    """
    threshold = float(strict_clean_threshold)

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "ml_strict_clean_threshold must be between 0.0 and 1.0."
        )

    required_bundle_keys = {
        "model",
        "target",
        "feature_columns",
        "cohort_filter",
    }
    missing_bundle_keys = sorted(
        required_bundle_keys.difference(model_bundle)
    )

    if missing_bundle_keys:
        raise KeyError(
            "ML quality model bundle is missing: "
            + ", ".join(missing_bundle_keys)
        )

    if model_bundle["target"] != "STRICT_CLEAN_LABEL":
        raise ValueError(
            "ML quality model must target STRICT_CLEAN_LABEL, got "
            f"{model_bundle['target']!r}."
        )

    if model_bundle["cohort_filter"] != "usable_only":
        raise ValueError(
            "ML quality model must be trained with cohort_filter='usable_only'."
        )

    required_label_columns = {
        "usable_label",
        "strict_clean_label",
    }
    missing_label_columns = sorted(
        required_label_columns.difference(df_features.columns)
    )

    if missing_label_columns:
        raise KeyError(
            "Pipeline features are missing ML label columns: "
            + ", ".join(missing_label_columns)
        )

    invalid_label_order = (
        (df_features["strict_clean_label"] == 1)
        & (df_features["usable_label"] != 1)
    )

    if invalid_label_order.any():
        raise ValueError(
            "STRICT_CLEAN_LABEL must be a subset of USABLE_LABEL before "
            "applying the ML quality model."
        )

    feature_columns = list(model_bundle["feature_columns"])

    if not feature_columns:
        raise ValueError("ML quality model contains no feature columns.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("ML quality model contains duplicate feature columns.")

    missing_feature_columns = [
        column
        for column in feature_columns
        if column not in df_features.columns
    ]

    if missing_feature_columns:
        raise KeyError(
            "Pipeline output is missing model features: "
            + ", ".join(missing_feature_columns)
        )

    model = model_bundle["model"]

    if not hasattr(model, "predict_proba"):
        raise TypeError("ML quality model does not support predict_proba().")

    model_classes = getattr(model, "classes_", None)

    if model_classes is None and hasattr(model, "steps"):
        model_classes = getattr(model.steps[-1][1], "classes_", None)

    if model_classes is None:
        raise AttributeError("ML quality model does not expose classes_.")

    model_classes = np.asarray(model_classes)
    class_zero_indices = np.flatnonzero(model_classes == 0)
    class_one_indices = np.flatnonzero(model_classes == 1)

    if (
        len(model_classes) != 2
        or len(class_zero_indices) != 1
        or len(class_one_indices) != 1
    ):
        raise ValueError(
            "ML quality model must contain exactly the binary classes 0 and 1; "
            f"got {model_classes.tolist()}."
        )

    for column in ML_QUALITY_PROBABILITY_COLUMNS:
        df_features[column] = np.nan

    usable_mask = df_features["usable_label"] == 1
    scored_windows = int(usable_mask.sum())

    summary = empty_ml_quality_summary(threshold)
    summary.update({
        "requested": True,
        "applied": True,
        "model_tag": str(model_bundle.get("model_tag", "")),
        "training_variant": str(
            model_bundle.get("training_variant", "")
        ),
        "feature_variant": str(model_bundle.get("feature_variant", "")),
        "cohort_filter": str(model_bundle["cohort_filter"]),
        "probability_calibrated": bool(
            model_bundle.get("probability_calibrated", False)
        ),
        "scored_windows": scored_windows,
    })

    if scored_windows == 0:
        return summary

    probabilities = np.asarray(
        model.predict_proba(
            df_features.loc[usable_mask, feature_columns]
        ),
        dtype=float,
    )

    if probabilities.shape != (scored_windows, len(model_classes)):
        raise ValueError(
            "Unexpected predict_proba() output shape: "
            f"{probabilities.shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("ML quality model returned non-finite probabilities.")

    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("ML quality model returned probabilities outside [0, 1].")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("ML quality probabilities do not sum to 1 per window.")

    p_usable_not_strict = probabilities[:, class_zero_indices[0]]
    p_strict_clean = probabilities[:, class_one_indices[0]]
    strict_clean_at_threshold = (
        p_strict_clean >= threshold
    ).astype(int)

    df_features.loc[
        usable_mask,
        "ML_P_USABLE_NOT_STRICT",
    ] = p_usable_not_strict
    df_features.loc[
        usable_mask,
        "ML_P_STRICT_CLEAN",
    ] = p_strict_clean
    df_features.loc[
        usable_mask,
        "ML_STRICT_CLEAN_AT_THRESHOLD",
    ] = strict_clean_at_threshold

    rule_usable_not_strict_mask = (
        usable_mask
        & (df_features["strict_clean_label"] == 0)
    )
    rule_strict_clean_mask = (
        usable_mask
        & (df_features["strict_clean_label"] == 1)
    )

    def mean_probability(column, mask):
        if not mask.any():
            return np.nan
        return float(df_features.loc[mask, column].mean())

    summary.update({
        "mean_p_usable_not_strict_all_usable": mean_probability(
            "ML_P_USABLE_NOT_STRICT",
            usable_mask,
        ),
        "mean_p_strict_clean_all_usable": mean_probability(
            "ML_P_STRICT_CLEAN",
            usable_mask,
        ),
        "mean_p_usable_not_strict_rule_usable_not_strict": mean_probability(
            "ML_P_USABLE_NOT_STRICT",
            rule_usable_not_strict_mask,
        ),
        "mean_p_strict_clean_rule_usable_not_strict": mean_probability(
            "ML_P_STRICT_CLEAN",
            rule_usable_not_strict_mask,
        ),
        "mean_p_usable_not_strict_rule_strict_clean": mean_probability(
            "ML_P_USABLE_NOT_STRICT",
            rule_strict_clean_mask,
        ),
        "mean_p_strict_clean_rule_strict_clean": mean_probability(
            "ML_P_STRICT_CLEAN",
            rule_strict_clean_mask,
        ),
        "strict_clean_windows_at_threshold": int(
            strict_clean_at_threshold.sum()
        ),
        "strict_clean_share_at_threshold": float(
            strict_clean_at_threshold.mean()
        ),
    })

    return summary


# END OPTIONAL V2 ML MODEL HELPERS


def apply_quality_labels(df_features):
    """Create QUALITY, USABLE and STRICT_CLEAN without changing definitions."""
    # =========================
    # WINDOW CLASSIFICATION
    # =========================

    threshold_corr = 0.3 # (Default: 0.3 slighty good windows, 0.5 good windows)
    threshold_reference = dict(
        df_features.attrs.get("ade_threshold_reference", {})
    )
    threshold_var_quality = threshold_reference.get(
        "threshold_var_quality"
    )

    if threshold_var_quality is None:
        threshold_var_quality = pooled_quantile( # (Default: 0.75 slightly strict, 0.85 softer)
            df_features,
            "var1",
            "var2",
            0.75
        )
        threshold_mode = "recording_adaptive"
        threshold_override_source = ""
    else:
        threshold_var_quality = float(threshold_var_quality)
        if not np.isfinite(threshold_var_quality):
            raise ValueError(
                "Referenced QUALITY_BASE variance threshold must be finite."
            )
        threshold_mode = "fixed_reference"
        threshold_override_source = str(
            threshold_reference.get("threshold_reference_file", "")
        )
    # -------------------------
    # QUALITY LABEL
    # -------------------------
    # Technically useful window via signal quality criteria
    # There can be still heuristically deteced artifact type despite quality_label=1

    # --- Quality without Autoreject as basis for comparison ---
    df_features["quality_base_label_eeg1"] = (
        (df_features["corr"].abs() > threshold_corr) &      # Channel similarity sufficient
        (df_features["var1"] < threshold_var_quality)       # Variance not too high
    ).astype(int) # 1 = teechnically viable, 0 = technically noticable/worse

    df_features["quality_base_label_eeg2"] = (
        (df_features["corr"].abs() > threshold_corr) &      # Channel similarity sufficient
        (df_features["var2"] < threshold_var_quality)       # Variance not too high
    ).astype(int) # 1 = teechnically viable, 0 = technically noticable/worse

    df_features["quality_base_label"] = (
        (df_features["quality_base_label_eeg1"] == 1) &
        (df_features["quality_base_label_eeg2"] == 1)
    ).astype(int)

    # --- Quality with Autoreject-Veto ---
    df_features["quality_label_eeg1"] = (
        (df_features["quality_base_label_eeg1"] == 1) &
        (~df_features["autoreject_bad_eeg1"])               # not rejected by Autoreject
    ).astype(int)

    df_features["quality_label_eeg2"] = (
        (df_features["quality_base_label_eeg2"] == 1) &
        (~df_features["autoreject_bad_eeg2"])               # not rejected by Autoreject
    ).astype(int)

    df_features["quality_label"] = (
        (df_features["quality_label_eeg1"] == 1) &
        (df_features["quality_label_eeg2"] == 1)
    ).astype(int)

    # -------------------------
    # USABLE LABEL
    # -------------------------
    # Technically viable and no hard artifact marker
    # SLOW_CORR may be included
    df_features["usable_label_eeg1"] = (
        (df_features["quality_label_eeg1"] == 1) &
        (df_features["has_no_hard_artifact_eeg1"])
    ).astype(int)

    df_features["usable_label_eeg2"] = (
        (df_features["quality_label_eeg2"] == 1) &
        (df_features["has_no_hard_artifact_eeg2"])
    ).astype(int)

    df_features["usable_label"] = (
        (df_features["usable_label_eeg1"] == 1) &
        (df_features["usable_label_eeg2"] == 1)
    ).astype(int)

    # -------------------------
    # STRICT CLEAN LABEL
    # -------------------------

    df_features["has_any_marker_eeg1"] = (
        df_features["has_slow_corr"] |
        df_features["has_emg_eeg1"] |
        df_features["has_ecg_eeg1"] |
        df_features["has_bad_auto_eeg1"] |
        df_features["has_wavelet_hf_eeg1"] |
        df_features["has_wavelet_transient_eeg1"] |
        df_features["has_amp_burst_eeg1"] |
        df_features["has_jump_eeg1"] |
        df_features["has_flatline_eeg1"] |
        df_features["has_hf_emg_eeg1"] |
        df_features["has_low_drift_eeg1"] |
        df_features["has_clipping_eeg1"]
    )

    df_features["has_any_marker_eeg2"] = (
        df_features["has_slow_corr"] |
        df_features["has_emg_eeg2"] |
        df_features["has_ecg_eeg2"] |
        df_features["has_bad_auto_eeg2"] |
        df_features["has_wavelet_hf_eeg2"] |
        df_features["has_wavelet_transient_eeg2"] |
        df_features["has_amp_burst_eeg2"] |
        df_features["has_jump_eeg2"] |
        df_features["has_flatline_eeg2"] |
        df_features["has_hf_emg_eeg2"] |
        df_features["has_low_drift_eeg2"] |
        df_features["has_clipping_eeg2"]
    )

    # Technically viable and absolutely no marker
    df_features["strict_clean_label_eeg1"] = (
        (df_features["quality_label_eeg1"] == 1) &
        (~df_features["has_any_marker_eeg1"])
    ).astype(int)

    df_features["strict_clean_label_eeg2"] = (
        (df_features["quality_label_eeg2"] == 1) &
        (~df_features["has_any_marker_eeg2"])
    ).astype(int)

    df_features["strict_clean_label"] = (
        (df_features["strict_clean_label_eeg1"] == 1) &
        (df_features["strict_clean_label_eeg2"] == 1)
    ).astype(int)

    return df_features, {
        "threshold_corr": threshold_corr,
        "threshold_var_quality": threshold_var_quality,
        "threshold_mode": threshold_mode,
        "threshold_override_source": threshold_override_source,
    }


def apply_ml_quality_assessment(
    *,
    df_features,
    apply_ml_quality_model,
    ml_quality_model_path,
    ml_strict_clean_threshold,
    selected_signal_method_success,
    signal_method,
    window_sec=None,
    window_overlap=None,
    sample_rate=None,
):
    # -------------------------
    # PROBABILISTIC QUALITY ASSESSMENT
    # -------------------------
    """
    The rule-based labels remain unchanged. The model only scores windows
    inside the usable cohort and distinguishes usable-not-strict (class 0)
    from strict-clean (class 1).
    """
    ml_quality_summary = empty_ml_quality_summary(
        ml_strict_clean_threshold
    )
    ml_quality_summary["requested"] = apply_ml_quality_model

    if apply_ml_quality_model and not selected_signal_method_success:
        print(
            "\nML quality assessment skipped: "
            f"signal_method='{signal_method}' was not successful."
        )

    elif apply_ml_quality_model:
        resolved_ml_model_path = (
            Path(ml_quality_model_path)
            .expanduser()
            .resolve()
        )
        ml_quality_model_bundle = load_ml_quality_model_bundle(
            str(resolved_ml_model_path)
        )
        application_context = describe_ml_application_context(
            ml_quality_model_bundle,
            signal_method,
            window_sec=window_sec,
            window_overlap=window_overlap,
            sample_rate=sample_rate,
        )
        ml_quality_summary = add_ml_quality_probabilities(
            df_features=df_features,
            model_bundle=ml_quality_model_bundle,
            strict_clean_threshold=ml_strict_clean_threshold,
        )
        ml_quality_summary["model_file"] = resolved_ml_model_path.name
        # The per-recording ml_summary sheet exports every summary field,
        # retaining this context warning alongside the numeric RF results.
        ml_quality_summary.update(application_context)
        if application_context["application_context_warning"]:
            warnings.warn(
                application_context["application_context_warning"],
                UserWarning,
                stacklevel=2,
            )

        print(
            "\nML quality windows scored:",
            ml_quality_summary["scored_windows"],
        )
        print(
            "ML mean p(strict_clean), all usable windows:",
            ml_quality_summary["mean_p_strict_clean_all_usable"],
        )
        print(
            "ML probability calibrated:",
            ml_quality_summary["probability_calibrated"],
        )
        print(
            "ML strict-clean windows at threshold "
            f"{ml_strict_clean_threshold:g}:",
            ml_quality_summary["strict_clean_windows_at_threshold"],
        )

    return ml_quality_summary


def form_quality_groups(df_features):
    # Create the existing compatibility groups and print their summaries
    # Keeping old label-column as alias, so older code-/plot-amounts do not break
    # label = quality_label
    df_features["label"] = df_features["quality_label"]

    df_exploded = df_features.explode("artifact_type")

    # -------------------------
    # Form groups
    # -------------------------
    df_quality_good = df_features[df_features["quality_label"] == 1]
    df_quality_bad = df_features[df_features["quality_label"] == 0]

    # Compatibility-alias for previous export-code
    df_good = df_quality_good

    df_good_usable= df_quality_good[df_quality_good["has_no_hard_artifact"]]
    df_good_strict_clean = df_quality_good[df_quality_good["is_strict_clean"]]

    df_good_slowcorr = df_quality_good[df_quality_good["has_slow_corr"]]
    df_good_emg = df_quality_good[df_quality_good["has_emg"]]
    df_good_ecg = df_quality_good[df_quality_good["has_ecg"]]
    df_good_bad_auto = df_quality_good[df_quality_good["has_bad_auto"]]
    df_good_mixed = df_quality_good[df_quality_good["is_mixed"]]

    df_good_wavelet_hf = df_quality_good[df_quality_good["has_wavelet_hf"]]
    df_good_wavelet_transient = df_quality_good[df_quality_good["has_wavelet_transient"]]

    df_good_amp_burst = df_quality_good[df_quality_good["has_amp_burst"]]
    df_good_jump = df_quality_good[df_quality_good["has_jump"]]
    df_good_flatline = df_quality_good[df_quality_good["has_flatline"]]
    df_good_hf_emg = df_quality_good[df_quality_good["has_hf_emg"]]
    df_good_low_drift = df_quality_good[df_quality_good["has_low_drift"]]
    df_good_clipping = df_quality_good[df_quality_good["has_clipping"]]

    # -------------------------
    # Output
    # -------------------------
    print("\nQuality-good windows:", int(df_features["quality_label"].sum()))
    print("Quality-bad windows:", int(len(df_features) - df_features["quality_label"].sum()))

    print("\nUsable windows:", int(df_features["usable_label"].sum()))
    print("Not usable windows:", int(len(df_features) - df_features["usable_label"].sum()))

    print("\nStrict-clean windows:", int(df_features["strict_clean_label"].sum()))
    print("Not strict-clean windows:", int(len(df_features) - df_features["strict_clean_label"].sum()))

    print("\nArtifact Summary:")
    print(df_exploded["artifact_type"].value_counts())

    print("\nMixed windows:", int(df_features["is_mixed"].sum()))

    print("\nQuality-good windows by artifact type:")
    print(df_exploded[df_exploded["quality_label"] == 1]["artifact_type"].value_counts())

    print("\nQuality-bad windows by artifact type:")
    print(df_exploded[df_exploded["quality_label"] == 0]["artifact_type"].value_counts())

    print("\nUsable / no-hard-artifact windows by marker:")
    print(df_exploded[df_exploded["usable_label"] == 1]["artifact_type"].value_counts())

    print("\nStrict-clean windows by marker:")
    print(df_exploded[df_exploded["strict_clean_label"] == 1]["artifact_type"].value_counts())

    return {
        "df_exploded": df_exploded,
        "df_quality_good": df_quality_good,
        "df_quality_bad": df_quality_bad,
        "df_good": df_good,
        "df_good_usable": df_good_usable,
        "df_good_strict_clean": df_good_strict_clean,
        "df_good_slowcorr": df_good_slowcorr,
        "df_good_emg": df_good_emg,
        "df_good_ecg": df_good_ecg,
        "df_good_bad_auto": df_good_bad_auto,
        "df_good_mixed": df_good_mixed,
        "df_good_wavelet_hf": df_good_wavelet_hf,
        "df_good_wavelet_transient": df_good_wavelet_transient,
        "df_good_amp_burst": df_good_amp_burst,
        "df_good_jump": df_good_jump,
        "df_good_flatline": df_good_flatline,
        "df_good_hf_emg": df_good_hf_emg,
        "df_good_low_drift": df_good_low_drift,
        "df_good_clipping": df_good_clipping,
    }
