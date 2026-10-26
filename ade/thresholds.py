"""Full-precision original thresholds for comparisons within one ADE batch.

The original is evaluated first. Its effective numeric thresholds are passed
in memory to other signal methods for the same recording. Existing marker
definitions and file-based reference interfaces remain in their own modules.
"""

import math
from pathlib import Path


THRESHOLD_EVALUATION_MODES = (
    "recording_adaptive",
    "fixed_original_thresholds",
)

THRESHOLD_OVERRIDE_PARAMETERS = {
    "autoreject_threshold_eeg": "autoreject_threshold_eeg_override",
    "threshold_var_emg": "threshold_var_emg_override",
    "threshold_wavelet_hf": "threshold_wavelet_hf_override",
    "threshold_wavelet_transient": "threshold_wavelet_transient_override",
    "threshold_amp_burst": "threshold_amp_burst_override",
    "threshold_jump": "threshold_jump_override",
    "threshold_flatline_std": "threshold_flatline_std_override",
    "threshold_flatline_ptp": "threshold_flatline_ptp_override",
    "threshold_hf_emg_line": "threshold_hf_emg_line_override",
    "threshold_low_drift_slope": "threshold_low_drift_slope_override",
}

PROCESSING_SETTINGS = (
    "use_dc", "use_detrend", "use_notch", "use_highpass", "use_lowpass",
    "use_normalization", "run_autoreject", "use_robust_artifact_markers",
    "use_wavelet_marker_features", "low_drift_is_hard_artifact",
    "wavelet_markers_are_hard_artifacts", "autoreject_threshold_eeg_override",
)


def validate_threshold_evaluation_mode(mode):
    if mode not in THRESHOLD_EVALUATION_MODES:
        raise ValueError(
            f"Unknown threshold_evaluation_mode={mode!r}; "
            f"choose {THRESHOLD_EVALUATION_MODES}."
        )
    return mode


def order_parameter_sets(parameter_sets, mode):
    """Put exactly one original first; add a reference run if necessary."""
    validate_threshold_evaluation_mode(mode)
    methods = [dict(parameters) for parameters in parameter_sets]
    if mode == "recording_adaptive" or not methods:
        return methods
    originals = [p for p in methods if p.get("signal_method", "original") == "original"]
    if len(originals) > 1:
        raise ValueError(
            "fixed_original_thresholds requires exactly one original "
            "parameter set per batch so its reference is unambiguous."
        )
    original = originals[0] if originals else {"signal_method": "original"}
    return [original] + [p for p in methods if p.get("signal_method", "original") != "original"]


def _comparison_settings(params):
    keys = set(PROCESSING_SETTINGS)
    keys.update(
        key for key in params
        if key.startswith("threshold_")
        and key != "threshold_evaluation_mode"
        and not key.startswith("threshold_reference_")
    )
    return {key: params.get(key) for key in sorted(keys)}


def make_original_threshold_reference(
    *, dataset, filename, params, fs, window_sec, window_overlap,
    number_of_windows, signal_lengths, artifact_thresholds,
    quality_thresholds, autoreject_result,
):
    """Capture the values actually used, before any spreadsheet round-trip."""
    if params.get("signal_method", "original") != "original":
        raise ValueError("Only an original run can provide comparison thresholds.")
    if number_of_windows <= 0:
        raise ValueError("An original reference must contain at least one window.")
    if params.get("run_autoreject", True) and not autoreject_result["autoreject_success"]:
        raise RuntimeError("Original AutoReject failed; fixed comparisons cannot use this reference.")
    values = {
        key: float(artifact_thresholds[key])
        for key in THRESHOLD_OVERRIDE_PARAMETERS
        if key != "autoreject_threshold_eeg"
    }
    values["threshold_var_quality"] = float(quality_thresholds["threshold_var_quality"])
    ar_threshold = autoreject_result["autoreject_threshold_eeg"]
    values["autoreject_threshold_eeg"] = float(ar_threshold) if ar_threshold is not None else None
    for name, value in values.items():
        if value is not None and math.isnan(value):
            raise ValueError(f"Original threshold {name} is NaN.")
    for name in ("threshold_var_emg", "threshold_var_quality"):
        if not math.isfinite(values[name]):
            raise ValueError(f"Original threshold {name} must be finite.")
    if params.get("run_autoreject", True) and (
        values["autoreject_threshold_eeg"] is None
        or not math.isfinite(values["autoreject_threshold_eeg"])
        or values["autoreject_threshold_eeg"] <= 0
    ):
        raise ValueError("Original AutoReject threshold must be finite and positive.")
    source = f"same-batch original {dataset}/{filename} (full precision)"
    if params.get("artifact_export_dir") and params.get("export_artifact_windows", False):
        export_path = Path(params["artifact_export_dir"]) / f"{filename}_artifacts.xlsx"
        source += f"; export={export_path}"
    return {
        "dataset": str(dataset), "filename": str(filename),
        "fs": float(fs), "window_sec": float(window_sec),
        "window_overlap": float(window_overlap),
        "number_of_windows": int(number_of_windows),
        "signal_lengths": tuple(map(int, signal_lengths)),
        "settings": _comparison_settings(params), "values": values,
        "source": source,
    }


def apply_original_threshold_reference(
    params, *, dataset, filename, fs, window_sec, window_overlap,
    number_of_windows, signal_lengths,
):
    """Inject existing numeric overrides and check the comparison context."""
    mode = validate_threshold_evaluation_mode(
        params.get("threshold_evaluation_mode", "recording_adaptive")
    )
    if mode == "recording_adaptive":
        return params, None
    if params.get("threshold_reference_artifact_file"):
        raise ValueError(
            "Choose either the same-batch fixed mode or a file-based "
            "threshold reference; do not mix both."
        )
    if params.get("signal_method", "original") == "original":
        return params, None
    reference = params.get("_original_threshold_reference")
    if not isinstance(reference, dict):
        raise RuntimeError(
            "No successful original reference from this recording is available. "
            "Run the original first; a failed reference is never replaced by an old export."
        )
    if (str(dataset), str(filename)) != (reference["dataset"], reference["filename"]):
        raise ValueError("Original threshold reference belongs to another recording.")
    actual = {"fs": float(fs), "window_sec": float(window_sec), "window_overlap": float(window_overlap)}
    for name, value in actual.items():
        if not math.isclose(value, reference[name], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Fixed comparisons require matching {name}: original={reference[name]}, candidate={value}.")
    if (
        number_of_windows != reference["number_of_windows"]
        or tuple(signal_lengths) != reference["signal_lengths"]
    ):
        raise ValueError("Fixed comparisons require matching window counts and signal lengths.")
    current = _comparison_settings(params)
    keys = set(reference["settings"]) | set(current)
    different = [key for key in sorted(keys) if current.get(key) != reference["settings"].get(key)]
    if different:
        raise ValueError(
            "Fixed comparisons require the same preprocessing and marker settings; differences: "
            + ", ".join(different)
        )
    prepared = dict(params)
    for value_name, parameter_name in THRESHOLD_OVERRIDE_PARAMETERS.items():
        value = reference["values"][value_name]
        # Disabled marker groups may deliberately use +/-infinity. Their
        # unchanged feature flags retain those sentinels in the marker code.
        if value is not None and math.isfinite(value):
            prepared[parameter_name] = value
    prepared["autoreject_threshold_override_source"] = reference["source"]
    print("Threshold evaluation mode: fixed_original_thresholds")
    print("Original threshold reference:", reference["source"])
    return prepared, reference


def attach_original_quality_reference(df_features, artifact_thresholds, reference):
    """Pass the frozen variance criterion through the existing label API."""
    if reference is None:
        return
    artifact_thresholds["threshold_mode"] = "fixed_reference"
    artifact_thresholds["threshold_override_source"] = reference["source"]
    df_features.attrs["ade_threshold_reference"] = {
        "threshold_var_quality": reference["values"]["threshold_var_quality"],
        "threshold_reference_file": reference["source"],
    }
    print("Artifact threshold mode: fixed_reference")
    print("Quality variance threshold mode: fixed_reference")


def check_applied_reference_thresholds(
    reference, artifact_thresholds, quality_thresholds, autoreject_result,
):
    """Fail explicitly if an installed detector ignored a numeric override."""
    if reference is None:
        return
    for name, expected in reference["values"].items():
        if name == "autoreject_threshold_eeg":
            actual = autoreject_result[name]
        elif name == "threshold_var_quality":
            actual = quality_thresholds[name]
        else:
            actual = artifact_thresholds[name]
        if actual != expected:
            raise RuntimeError(
                f"Fixed original threshold was not applied: {name}; "
                f"original={expected}, actual={actual}. Check the installed "
                "ade/artifacts.py and ade/quality.py reference interfaces."
            )
