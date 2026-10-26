"""Self-contained v2.5 LOW_DRIFT hybrid selection and production inference.

The tested selection, preservation gates and metadata are retained from the
development implementation. No development runner or cohort split is imported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

BASE_POLICY_VERSION = "v2.3"
HYBRID_POLICY_VERSION = "v2.5"
CHANNEL_MARKERS = (
    "EMG", "ECG", "BAD_AUTO", "WAVELET_HF", "WAVELET_TRANSIENT",
    "AMP_BURST", "JUMP", "FLATLINE", "HF_EMG", "LOW_DRIFT", "CLIPPING",
)
CORRELATION_EPSILON = 1e-15
POWER_EPSILON = 1e-30



def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]).copy() for name in saved.files}


def scalar_text(payload: dict[str, np.ndarray], key: str) -> str | None:
    if key not in payload:
        return None
    value = np.asarray(payload[key])
    if value.ndim != 0:
        raise ValueError(f"NPZ field {key!r} must be scalar.")
    return str(value.item())


def require_keys(
    payload: dict[str, np.ndarray],
    required: Iterable[str],
    path: Path,
) -> None:
    missing = sorted(set(required).difference(payload))
    if missing:
        raise KeyError(path.name + " is missing: " + ", ".join(missing))


def read_windows(path: Path, expected_windows: int) -> pd.DataFrame:
    frame = pd.read_excel(
        path,
        sheet_name="windows",
        engine="openpyxl",
        index_col=0,
    )
    if len(frame) != expected_windows:
        raise ValueError(
            f"{path.name} contains {len(frame)} windows; "
            f"expected {expected_windows}."
        )
    numeric_index = pd.to_numeric(frame.index, errors="raise").to_numpy(int)
    zero_based = np.arange(expected_windows)
    one_based = np.arange(1, expected_windows + 1)
    if not (
        np.array_equal(numeric_index, zero_based)
        or np.array_equal(numeric_index, one_based)
    ):
        raise ValueError(
            f"{path.name} window index is neither 0..N-1 nor 1..N."
        )
    frame = frame.reset_index(drop=True)
    return frame


def binary_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise KeyError(f"Artifact workbook is missing column {name!r}.")
    values = pd.to_numeric(frame[name], errors="raise").to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"Artifact column {name!r} contains non-binary values.")
    return values.astype(bool, copy=False)


def channel_marker_tensor(frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros((len(frame), 2, len(CHANNEL_MARKERS)), dtype=bool)
    for channel in range(2):
        suffix = f"EEG{channel + 1}"
        for marker_index, marker in enumerate(CHANNEL_MARKERS):
            result[:, channel, marker_index] = binary_column(
                frame, f"{marker}_{suffix}"
            )
    return result


def validate_artifact_alignment(
    original: pd.DataFrame,
    candidate: pd.DataFrame,
    recording_id: str,
) -> None:
    for column in ("begin_idx", "end_idx"):
        if column not in original or column not in candidate:
            raise KeyError(f"Artifact workbooks are missing {column!r}.")
        before = pd.to_datetime(original[column], errors="raise").to_numpy()
        after = pd.to_datetime(candidate[column], errors="raise").to_numpy()
        if not np.array_equal(before, after):
            raise ValueError(
                f"Artifact-window {column} mismatch for {recording_id}."
            )


def row_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    numerator = np.sum(first * second, axis=1)
    denominator = np.sqrt(
        np.sum(first**2, axis=1) * np.sum(second**2, axis=1)
    )
    result = np.full(len(first), np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > CORRELATION_EPSILON,
    )
    return np.clip(result, -1.0, 1.0)


def row_rrmse(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    error = np.sqrt(np.mean(np.square(candidate - reference), axis=1))
    centered = reference - reference.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(np.square(centered), axis=1))
    result = np.full(len(reference), np.nan, dtype=np.float64)
    np.divide(
        error,
        scale,
        out=result,
        where=scale > CORRELATION_EPSILON,
    )
    return result


def highpass_rows(
    values: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(
            f"Invalid high-pass cutoff {cutoff_hz:g} Hz for fs={sample_rate:g}."
        )
    sections = butter(
        4,
        cutoff_hz,
        btype="highpass",
        fs=sample_rate,
        output="sos",
    )
    return sosfiltfilt(
        sections,
        np.asarray(values, dtype=np.float64),
        axis=1,
    )


def trim_rows(values: np.ndarray, trim_samples: int) -> np.ndarray:
    if trim_samples <= 0:
        return values
    if values.shape[1] <= 2 * trim_samples + 8:
        raise ValueError(
            f"Edge trim {trim_samples} leaves too few of "
            f"{values.shape[1]} samples."
        )
    return values[:, trim_samples:-trim_samples]


def marker_names(mask: np.ndarray) -> str:
    return "|".join(
        marker for marker, active in zip(CHANNEL_MARKERS, mask) if active
    )


def _validate_base_numeric_args(args: argparse.Namespace) -> None:
    if not -1.0 <= args.preservation_correlation_min <= 1.0:
        raise ValueError("--preservation-correlation-min must be within [-1, 1].")
    if args.preservation_rrmse_max < 0.0:
        raise ValueError("--preservation-rrmse-max must be non-negative.")
    if args.low_drift_highpass_hz <= 0.0:
        raise ValueError("--low-drift-highpass-hz must be positive.")
    if args.edge_trim_sec < 0.0:
        raise ValueError("--edge-trim-sec must be non-negative.")


def validate_source_alignment(
    recording_id: str,
    v21: dict[str, np.ndarray],
    v3: dict[str, np.ndarray],
    v21_path: Path,
    v3_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v21_required = {"recording_id", "window_number", "eeg", "denoised_eeg"}
    v3_required = {
        "recording_id",
        "sample_rate",
        "window_number",
        "eeg",
        "denoised_eeg",
        "inference_policy_version",
        "inference_policy_json",
        "inference_processed_eeg",
        "inference_preserved_strict_clean_eeg",
        "inference_bypassed_marker_eeg",
        "inference_low_information_bypassed_eeg",
        "inference_output_safety_checked_eeg",
        "inference_output_safety_fallback_eeg",
        "inference_output_safety_new_flatline_eeg",
        "inference_output_safety_new_clipping_eeg",
        "inference_selective_safety_checked_eeg",
        "inference_selective_safety_fallback_eeg",
        "inference_selective_low_drift_checked_eeg",
        "inference_selective_low_drift_failed_eeg",
        "inference_selective_event_checked_eeg",
        "inference_selective_event_failed_eeg",
        "inference_selective_ecg_checked_eeg",
        "inference_selective_ecg_failed_eeg",
    }
    require_keys(v21, v21_required, v21_path)
    require_keys(v3, v3_required, v3_path)
    for payload, path in ((v21, v21_path), (v3, v3_path)):
        stored = scalar_text(payload, "recording_id")
        if stored != recording_id:
            raise ValueError(
                f"{path.name} belongs to {stored!r}, expected {recording_id!r}."
            )
    v3_policy = scalar_text(v3, "inference_policy_version")
    if v3_policy != BASE_POLICY_VERSION:
        raise ValueError(
            f"{v3_path.name} uses policy {v3_policy!r}; "
            f"expected {BASE_POLICY_VERSION!r}."
        )
    v21_policy = scalar_text(v21, "inference_policy_version")
    if v21_policy is not None and v21_policy != "v2.1":
        raise ValueError(
            f"{v21_path.name} uses policy {v21_policy!r}; expected 'v2.1'."
        )

    source_v21 = np.asarray(v21["eeg"], dtype=np.float32)
    source_v3 = np.asarray(v3["eeg"], dtype=np.float32)
    output_v21 = np.asarray(v21["denoised_eeg"], dtype=np.float32)
    output_v3 = np.asarray(v3["denoised_eeg"], dtype=np.float32)
    if source_v21.ndim != 3 or source_v21.shape[1] != 2:
        raise ValueError(f"Unexpected EEG shape in {v21_path}: {source_v21.shape}")
    expected_shape = source_v21.shape
    for name, values in (
        ("v3 source", source_v3),
        ("v2.1 output", output_v21),
        ("v3-selective output", output_v3),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {expected_shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite samples.")
    if not np.array_equal(v21["window_number"], v3["window_number"]):
        raise ValueError(f"Prediction window-number mismatch for {recording_id}.")
    if not np.allclose(source_v21, source_v3, rtol=0.0, atol=1e-6):
        difference = float(np.max(np.abs(source_v21 - source_v3)))
        raise ValueError(
            f"Prediction source mismatch for {recording_id}; "
            f"maximum difference={difference:.6g}."
        )
    return source_v3, output_v21, output_v3, np.asarray(
        v3["window_number"], dtype=np.int64
    )


def _build_base_candidate(
    *,
    recording_id: str,
    v21_path: Path,
    v3_path: Path,
    original_artifact_path: Path,
    v21_artifact_path: Path,
    args: argparse.Namespace,
    policy: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, int]]:
    v21 = load_npz(v21_path)
    v3 = load_npz(v3_path)
    source, output_v21, output_v3, window_numbers = validate_source_alignment(
        recording_id,
        v21,
        v3,
        v21_path,
        v3_path,
    )
    windows, channels, samples = source.shape
    decision_shape = (windows, channels)
    sample_rate = float(np.asarray(v3["sample_rate"]).item())
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"Invalid sample rate in {v3_path}.")

    original_frame = read_windows(original_artifact_path, windows)
    v21_frame = read_windows(v21_artifact_path, windows)
    validate_artifact_alignment(original_frame, v21_frame, recording_id)
    original_markers = channel_marker_tensor(original_frame)
    v21_markers = channel_marker_tensor(v21_frame)
    low_drift_index = CHANNEL_MARKERS.index("LOW_DRIFT")

    processed = np.asarray(v3["inference_processed_eeg"], dtype=bool)
    output_safety_fallback = np.asarray(
        v3["inference_output_safety_fallback_eeg"], dtype=bool
    )
    v3_low_failed = np.asarray(
        v3["inference_selective_low_drift_failed_eeg"], dtype=bool
    )
    v3_event_failed = np.asarray(
        v3["inference_selective_event_failed_eeg"], dtype=bool
    )
    v3_ecg_failed = np.asarray(
        v3["inference_selective_ecg_failed_eeg"], dtype=bool
    )
    for name, values in (
        ("processed", processed),
        ("output_safety_fallback", output_safety_fallback),
        ("low_drift_failed", v3_low_failed),
        ("event_failed", v3_event_failed),
        ("ecg_failed", v3_ecg_failed),
    ):
        if values.shape != decision_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {decision_shape}."
            )

    v21_processed = np.ones(decision_shape, dtype=bool)
    if "inference_processed_eeg" in v21:
        v21_processed = np.asarray(v21["inference_processed_eeg"], dtype=bool)
        if v21_processed.shape != decision_shape:
            raise ValueError("v2.1 processed mask has an incompatible shape.")

    flattened_source = source.reshape(-1, samples)
    flattened_v21 = output_v21.reshape(-1, samples)
    general_correlation = row_correlation(flattened_source, flattened_v21).reshape(
        decision_shape
    )
    general_rrmse = row_rrmse(flattened_source, flattened_v21).reshape(
        decision_shape
    )

    filtered_source = highpass_rows(
        flattened_source,
        sample_rate,
        args.low_drift_highpass_hz,
    )
    filtered_v21 = highpass_rows(
        flattened_v21,
        sample_rate,
        args.low_drift_highpass_hz,
    )
    trim_samples = int(round(args.edge_trim_sec * sample_rate))
    filtered_source = trim_rows(filtered_source, trim_samples)
    filtered_v21 = trim_rows(filtered_v21, trim_samples)
    low_correlation = row_correlation(filtered_source, filtered_v21).reshape(
        decision_shape
    )
    low_rrmse = row_rrmse(filtered_source, filtered_v21).reshape(decision_shape)

    original_low_drift = original_markers[:, :, low_drift_index]
    v21_low_drift = v21_markers[:, :, low_drift_index]
    low_drift_resolved = original_low_drift & ~v21_low_drift
    introduced = ~original_markers & v21_markers
    introduced_count = introduced.sum(axis=2)
    marker_gate = low_drift_resolved & (introduced_count == 0)
    general_gate = (
        np.isfinite(general_correlation)
        & np.isfinite(general_rrmse)
        & (general_correlation >= args.preservation_correlation_min)
        & (general_rrmse <= args.preservation_rrmse_max)
    )
    low_drift_gate = (
        np.isfinite(low_correlation)
        & np.isfinite(low_rrmse)
        & (low_correlation >= args.preservation_correlation_min)
        & (low_rrmse <= args.preservation_rrmse_max)
    )

    candidate = v3_low_failed & original_low_drift
    exclusive_failure = ~v3_event_failed & ~v3_ecg_failed
    eligible_state = processed & ~output_safety_fallback & v21_processed
    selected = (
        candidate
        & exclusive_failure
        & eligible_state
        & marker_gate
        & general_gate
        & low_drift_gate
    )

    hybrid_output = output_v3.copy()
    hybrid_output[selected] = output_v21[selected]
    payload = {name: np.asarray(value).copy() for name, value in v3.items()}
    payload["denoised_eeg"] = hybrid_output.astype(np.float32)
    difference = hybrid_output - source
    payload["input_to_output_rmse"] = np.sqrt(
        np.mean(np.square(difference), axis=2)
    ).astype(np.float32)

    low_failed = v3_low_failed.copy()
    event_failed = v3_event_failed.copy()
    ecg_failed = v3_ecg_failed.copy()
    low_failed[selected] = False
    selective_fallback = low_failed | event_failed | ecg_failed
    payload["inference_selective_low_drift_failed_eeg"] = low_failed.astype(
        np.int8
    )
    payload["inference_selective_event_failed_eeg"] = event_failed.astype(np.int8)
    payload["inference_selective_ecg_failed_eeg"] = ecg_failed.astype(np.int8)
    payload["inference_selective_safety_fallback_eeg"] = (
        selective_fallback.astype(np.int8)
    )
    if "inference_selective_low_drift_highpass_correlation_eeg" in payload:
        values = np.asarray(
            payload["inference_selective_low_drift_highpass_correlation_eeg"],
            dtype=np.float32,
        ).copy()
        values[selected] = low_correlation[selected]
        payload["inference_selective_low_drift_highpass_correlation_eeg"] = values
    if "inference_selective_low_drift_highpass_rrmse_eeg" in payload:
        values = np.asarray(
            payload["inference_selective_low_drift_highpass_rrmse_eeg"],
            dtype=np.float32,
        ).copy()
        values[selected] = low_rrmse[selected]
        payload["inference_selective_low_drift_highpass_rrmse_eeg"] = values
    payload["inference_selective_safety_fallback_count"] = np.array(
        int(selective_fallback.sum()), dtype=np.int64
    )
    payload["inference_selective_low_drift_failed_count"] = np.array(
        int(low_failed.sum()), dtype=np.int64
    )
    payload["inference_selective_event_failed_count"] = np.array(
        int(event_failed.sum()), dtype=np.int64
    )
    payload["inference_selective_ecg_failed_count"] = np.array(
        int(ecg_failed.sum()), dtype=np.int64
    )

    policy_json = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    payload.update(
        {
            "hybrid_policy_version": np.array(HYBRID_POLICY_VERSION),
            "hybrid_policy_json": np.array(policy_json),
            "hybrid_base_prediction_file": np.array(str(v3_path)),
            "hybrid_v21_prediction_file": np.array(str(v21_path)),
            "hybrid_original_artifact_file": np.array(
                str(original_artifact_path)
            ),
            "hybrid_v21_artifact_file": np.array(str(v21_artifact_path)),
            "hybrid_v21_candidate_eeg": candidate.astype(np.int8),
            "hybrid_v21_selected_eeg": selected.astype(np.int8),
            "hybrid_v21_marker_gate_pass_eeg": marker_gate.astype(np.int8),
            "hybrid_v21_general_preservation_pass_eeg": general_gate.astype(
                np.int8
            ),
            "hybrid_v21_low_drift_preservation_pass_eeg": (
                low_drift_gate.astype(np.int8)
            ),
            "hybrid_v21_low_drift_resolved_eeg": low_drift_resolved.astype(
                np.int8
            ),
            "hybrid_v21_introduced_marker_count_eeg": introduced_count.astype(
                np.int16
            ),
            "hybrid_v21_input_output_correlation_eeg": (
                general_correlation.astype(np.float32)
            ),
            "hybrid_v21_temporal_rrmse_eeg": general_rrmse.astype(np.float32),
            "hybrid_v21_low_drift_highpass_correlation_eeg": (
                low_correlation.astype(np.float32)
            ),
            "hybrid_v21_low_drift_highpass_rrmse_eeg": (
                low_rrmse.astype(np.float32)
            ),
            "hybrid_v21_candidate_count": np.array(
                int(candidate.sum()), dtype=np.int64
            ),
            "hybrid_v21_selected_count": np.array(
                int(selected.sum()), dtype=np.int64
            ),
        }
    )

    rows: list[dict[str, Any]] = []
    candidate_indices = np.argwhere(candidate)
    for window_index, channel_index in candidate_indices:
        reasons: list[str] = []
        if v3_event_failed[window_index, channel_index]:
            reasons.append("simultaneous_event_failure")
        if v3_ecg_failed[window_index, channel_index]:
            reasons.append("simultaneous_ecg_failure")
        if not processed[window_index, channel_index]:
            reasons.append("not_processed_by_v3")
        if output_safety_fallback[window_index, channel_index]:
            reasons.append("v3_output_safety_fallback")
        if not v21_processed[window_index, channel_index]:
            reasons.append("not_processed_by_v21")
        if not low_drift_resolved[window_index, channel_index]:
            reasons.append("low_drift_not_resolved")
        introduced_here = introduced[window_index, channel_index]
        if introduced_here.any():
            reasons.append("new_channel_marker")
        if not general_gate[window_index, channel_index]:
            reasons.append("general_preservation_failed")
        if not low_drift_gate[window_index, channel_index]:
            reasons.append("low_drift_preservation_failed")
        is_selected = bool(selected[window_index, channel_index])
        if is_selected and reasons:
            raise AssertionError("Selected hybrid row contains rejection reasons.")
        original_here = original_markers[window_index, channel_index]
        v21_here = v21_markers[window_index, channel_index]
        rows.append(
            {
                "recording_id": recording_id,
                "window_number": int(window_numbers[window_index]),
                "artifact_row_number": int(window_index + 1),
                "channel": f"EEG{channel_index + 1}",
                "decision": "use_v21" if is_selected else "keep_v3select",
                "selected_v21": is_selected,
                "rejection_reasons": "|".join(reasons),
                "original_channel_markers": marker_names(original_here),
                "v21_channel_markers": marker_names(v21_here),
                "resolved_channel_markers": marker_names(
                    original_here & ~v21_here
                ),
                "introduced_channel_markers": marker_names(introduced_here),
                "v3_event_failed": bool(
                    v3_event_failed[window_index, channel_index]
                ),
                "v3_ecg_failed": bool(
                    v3_ecg_failed[window_index, channel_index]
                ),
                "v21_general_correlation": float(
                    general_correlation[window_index, channel_index]
                ),
                "v21_general_rrmse": float(
                    general_rrmse[window_index, channel_index]
                ),
                "v21_low_drift_highpass_correlation": float(
                    low_correlation[window_index, channel_index]
                ),
                "v21_low_drift_highpass_rrmse": float(
                    low_rrmse[window_index, channel_index]
                ),
                "marker_gate_pass": bool(
                    marker_gate[window_index, channel_index]
                ),
                "general_preservation_pass": bool(
                    general_gate[window_index, channel_index]
                ),
                "low_drift_preservation_pass": bool(
                    low_drift_gate[window_index, channel_index]
                ),
            }
        )

    counts = {
        "channel_windows": int(windows * channels),
        "v3_selective_fallback": int(
            np.asarray(
                v3["inference_selective_safety_fallback_eeg"], dtype=bool
            ).sum()
        ),
        "low_drift_candidates": int(candidate.sum()),
        "hybrid_v21_selected": int(selected.sum()),
        "hybrid_v21_rejected": int(candidate.sum() - selected.sum()),
    }
    return payload, rows, counts


def _validate_base_payload(
    payload: dict[str, np.ndarray],
    expected_policy: dict[str, Any],
    path: Path,
) -> None:
    required = {
        "eeg",
        "denoised_eeg",
        "inference_processed_eeg",
        "inference_preserved_strict_clean_eeg",
        "inference_bypassed_marker_eeg",
        "inference_low_information_bypassed_eeg",
        "inference_output_safety_fallback_eeg",
        "inference_selective_safety_checked_eeg",
        "inference_selective_safety_fallback_eeg",
        "inference_selective_low_drift_checked_eeg",
        "inference_selective_low_drift_failed_eeg",
        "inference_selective_event_checked_eeg",
        "inference_selective_event_failed_eeg",
        "inference_selective_ecg_checked_eeg",
        "inference_selective_ecg_failed_eeg",
        "hybrid_policy_version",
        "hybrid_policy_json",
        "hybrid_v21_selected_eeg",
    }
    require_keys(payload, required, path)
    if scalar_text(payload, "inference_policy_version") != BASE_POLICY_VERSION:
        raise ValueError(f"{path.name} lost its v2.3 bridge compatibility.")
    if scalar_text(payload, "hybrid_policy_version") != HYBRID_POLICY_VERSION:
        raise ValueError(f"{path.name} has an unexpected hybrid policy.")
    stored_policy = json.loads(scalar_text(payload, "hybrid_policy_json") or "{}")
    if stored_policy != expected_policy:
        raise ValueError(f"{path.name} hybrid policy parameters do not match.")

    source = np.asarray(payload["eeg"], dtype=np.float32)
    output = np.asarray(payload["denoised_eeg"], dtype=np.float32)
    shape = source.shape[:2]
    if source.ndim != 3 or source.shape[1] != 2 or output.shape != source.shape:
        raise ValueError(f"Invalid EEG shape in {path}: {source.shape}.")
    preserved = np.asarray(
        payload["inference_preserved_strict_clean_eeg"], dtype=bool
    )
    bypassed = np.asarray(payload["inference_bypassed_marker_eeg"], dtype=bool)
    low_information = np.asarray(
        payload["inference_low_information_bypassed_eeg"], dtype=bool
    )
    safety = np.asarray(
        payload["inference_output_safety_fallback_eeg"], dtype=bool
    )
    low_failed = np.asarray(
        payload["inference_selective_low_drift_failed_eeg"], dtype=bool
    )
    event_failed = np.asarray(
        payload["inference_selective_event_failed_eeg"], dtype=bool
    )
    ecg_failed = np.asarray(
        payload["inference_selective_ecg_failed_eeg"], dtype=bool
    )
    selective = np.asarray(
        payload["inference_selective_safety_fallback_eeg"], dtype=bool
    )
    selected = np.asarray(payload["hybrid_v21_selected_eeg"], dtype=bool)
    for name, values in (
        ("preserved", preserved),
        ("bypassed", bypassed),
        ("low_information", low_information),
        ("safety", safety),
        ("selective", selective),
        ("selected", selected),
    ):
        if values.shape != shape:
            raise ValueError(f"{path.name} {name} mask has shape {values.shape}.")
    if not np.array_equal(selective, low_failed | event_failed | ecg_failed):
        raise ValueError(f"{path.name} has inconsistent selective masks.")
    exact_copy = preserved | bypassed | low_information | safety | selective
    if not np.array_equal(output[exact_copy], source[exact_copy]):
        raise ValueError(f"{path.name} violates an exact-copy fallback.")
    if np.any(selected & exact_copy):
        raise ValueError(f"{path.name} selects v2.1 inside an exact-copy state.")


def atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def hybrid_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": HYBRID_POLICY_VERSION,
        "base_prediction": "v3 checkpoint with inference policy v2.3",
        "substitution_prediction": "v2 checkpoint with inference policy v2.1",
        "enabled_marker": "LOW_DRIFT",
        "inherits_v24_gates": True,
        "require_same_channel_marker_resolution": True,
        "allow_new_channel_markers": False,
        "allow_simultaneous_event_or_ecg_failure": False,
        "preservation_correlation_min": float(
            args.preservation_correlation_min
        ),
        "preservation_rrmse_max": float(args.preservation_rrmse_max),
        "low_drift_highpass_hz": float(args.low_drift_highpass_hz),
        "edge_trim_sec": float(args.edge_trim_sec),
        "absolute_power_band_hz": [4.0, 30.0],
        "absolute_power_ratio_min": float(args.absolute_power_4_30_min),
        "block_new_slow_corr": True,
        "slow_corr_delta_rel_threshold": float(
            args.slow_corr_delta_rel_threshold
        ),
        "slow_corr_abs_correlation_threshold": float(
            args.slow_corr_correlation_threshold
        ),
        "screening_status": "exploratory_not_clinically_validated",
    }


def validate_numeric_args(args: argparse.Namespace) -> None:
    _validate_base_numeric_args(args)
    if args.absolute_power_4_30_min < 0.0:
        raise ValueError("--absolute-power-4-30-min must be non-negative.")
    if not 0.0 <= args.slow_corr_delta_rel_threshold <= 1.0:
        raise ValueError(
            "--slow-corr-delta-rel-threshold must be within [0, 1]."
        )
    if not 0.0 <= args.slow_corr_correlation_threshold <= 1.0:
        raise ValueError(
            "--slow-corr-correlation-threshold must be within [0, 1]."
        )


def absolute_power_ratio_4_30(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """Return absolute 4-30 Hz output/input power using the audit FFT method."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("Power-ratio inputs must be equally shaped 2-D arrays.")
    samples = reference.shape[1]
    frequencies = np.fft.rfftfreq(samples, d=1.0 / sample_rate)
    frequency_mask = (frequencies >= 4.0) & (frequencies <= 30.0)
    if frequency_mask.sum() < 2:
        raise ValueError(
            f"No usable 4-30 Hz bins for sample rate {sample_rate:g}."
        )
    taper = np.hanning(samples)[None, :]
    reference_centered = reference - reference.mean(axis=1, keepdims=True)
    candidate_centered = candidate - candidate.mean(axis=1, keepdims=True)
    reference_power = np.abs(
        np.fft.rfft(reference_centered * taper, axis=1)
    ) ** 2
    candidate_power = np.abs(
        np.fft.rfft(candidate_centered * taper, axis=1)
    ) ** 2
    reference_total = reference_power[:, frequency_mask].sum(axis=1)
    candidate_total = candidate_power[:, frequency_mask].sum(axis=1)
    ratio = np.full(len(reference), np.nan, dtype=np.float64)
    np.divide(
        candidate_total,
        reference_total,
        out=ratio,
        where=reference_total > POWER_EPSILON,
    )
    return ratio


def _welch_bandpower_rows(
    values: np.ndarray,
    sample_rate: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Welch bandpower input must be a 2-D array.")
    nperseg = min(256, values.shape[1] // 2)
    frequencies, spectrum = welch(
        values,
        fs=sample_rate,
        nperseg=nperseg,
        axis=1,
    )
    mask = (frequencies >= lower) & (frequencies <= upper)
    if mask.sum() < 2:
        raise ValueError(
            f"No usable {lower:g}-{upper:g} Hz bins for fs={sample_rate:g}."
        )
    return np.trapezoid(
        spectrum[:, mask],
        frequencies[mask],
        axis=1,
    )


def slow_corr_mask(
    eeg: np.ndarray,
    sample_rate: float,
    delta_relative_threshold: float,
    correlation_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce ADE's combined-window SLOW_CORR rule exactly."""
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 3 or eeg.shape[1] != 2:
        raise ValueError(f"Expected EEG shape (windows, 2, samples), got {eeg.shape}.")
    delta_relative: list[np.ndarray] = []
    for channel in range(2):
        values = eeg[:, channel, :]
        delta = _welch_bandpower_rows(values, sample_rate, 0.5, 4.0)
        total = _welch_bandpower_rows(values, sample_rate, 0.5, 30.0)
        safe_total = total.copy()
        safe_total[safe_total == 0.0] = 1e-10
        delta_relative.append(delta / safe_total)
    delta_relative_average = (delta_relative[0] + delta_relative[1]) / 2.0
    correlation = row_correlation(eeg[:, 0, :], eeg[:, 1, :])
    marker = (
        (delta_relative_average > delta_relative_threshold)
        & (np.abs(correlation) > correlation_threshold)
    )
    return marker, delta_relative_average, correlation


def _restore_selective_masks(
    payload: dict[str, np.ndarray],
    v3: dict[str, np.ndarray],
    selected: np.ndarray,
    low_correlation: np.ndarray,
    low_rrmse: np.ndarray,
) -> None:
    low_failed = np.asarray(
        v3["inference_selective_low_drift_failed_eeg"], dtype=bool
    ).copy()
    event_failed = np.asarray(
        v3["inference_selective_event_failed_eeg"], dtype=bool
    ).copy()
    ecg_failed = np.asarray(
        v3["inference_selective_ecg_failed_eeg"], dtype=bool
    ).copy()
    low_failed[selected] = False
    selective_fallback = low_failed | event_failed | ecg_failed
    payload["inference_selective_low_drift_failed_eeg"] = low_failed.astype(
        np.int8
    )
    payload["inference_selective_event_failed_eeg"] = event_failed.astype(
        np.int8
    )
    payload["inference_selective_ecg_failed_eeg"] = ecg_failed.astype(np.int8)
    payload["inference_selective_safety_fallback_eeg"] = (
        selective_fallback.astype(np.int8)
    )
    if "inference_selective_low_drift_highpass_correlation_eeg" in v3:
        values = np.asarray(
            v3["inference_selective_low_drift_highpass_correlation_eeg"],
            dtype=np.float32,
        ).copy()
        values[selected] = low_correlation[selected]
        payload[
            "inference_selective_low_drift_highpass_correlation_eeg"
        ] = values
    if "inference_selective_low_drift_highpass_rrmse_eeg" in v3:
        values = np.asarray(
            v3["inference_selective_low_drift_highpass_rrmse_eeg"],
            dtype=np.float32,
        ).copy()
        values[selected] = low_rrmse[selected]
        payload["inference_selective_low_drift_highpass_rrmse_eeg"] = values
    payload["inference_selective_safety_fallback_count"] = np.array(
        int(selective_fallback.sum()), dtype=np.int64
    )
    payload["inference_selective_low_drift_failed_count"] = np.array(
        int(low_failed.sum()), dtype=np.int64
    )
    payload["inference_selective_event_failed_count"] = np.array(
        int(event_failed.sum()), dtype=np.int64
    )
    payload["inference_selective_ecg_failed_count"] = np.array(
        int(ecg_failed.sum()), dtype=np.int64
    )


def build_hybrid(
    *,
    recording_id: str,
    v21_path: Path,
    v3_path: Path,
    original_artifact_path: Path,
    v21_artifact_path: Path,
    args: argparse.Namespace,
    policy: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, int]]:
    """Apply v2.5's two additional gates to the proven v2.4 selection."""
    payload, rows, counts = _build_base_candidate(
        recording_id=recording_id,
        v21_path=v21_path,
        v3_path=v3_path,
        original_artifact_path=original_artifact_path,
        v21_artifact_path=v21_artifact_path,
        args=args,
        policy=policy,
    )
    v21 = load_npz(v21_path)
    v3 = load_npz(v3_path)
    source = np.asarray(v3["eeg"], dtype=np.float32)
    output_v21 = np.asarray(v21["denoised_eeg"], dtype=np.float32)
    output_v3 = np.asarray(v3["denoised_eeg"], dtype=np.float32)
    windows, channels, samples = source.shape
    decision_shape = (windows, channels)
    sample_rate = float(np.asarray(v3["sample_rate"]).item())

    v24_selected = np.asarray(
        payload["hybrid_v21_selected_eeg"], dtype=bool
    ).copy()
    power_ratio = absolute_power_ratio_4_30(
        source.reshape(-1, samples),
        output_v21.reshape(-1, samples),
        sample_rate,
    ).reshape(decision_shape)
    power_gate = (
        np.isfinite(power_ratio)
        & (power_ratio >= args.absolute_power_4_30_min)
    )

    # First apply the power gate. SLOW_CORR is then evaluated on the actual
    # two-channel window that these still-eligible substitutions would create.
    power_selected = v24_selected & power_gate
    tentative_output = output_v3.copy()
    tentative_output[power_selected] = output_v21[power_selected]
    v3_slow_corr, v3_delta_relative, v3_channel_correlation = slow_corr_mask(
        output_v3,
        sample_rate,
        args.slow_corr_delta_rel_threshold,
        args.slow_corr_correlation_threshold,
    )
    tentative_slow_corr, tentative_delta_relative, tentative_correlation = (
        slow_corr_mask(
            tentative_output,
            sample_rate,
            args.slow_corr_delta_rel_threshold,
            args.slow_corr_correlation_threshold,
        )
    )
    new_slow_corr = tentative_slow_corr & ~v3_slow_corr
    slow_corr_gate = np.broadcast_to(
        (~new_slow_corr)[:, None], decision_shape
    ).copy()
    selected = power_selected & slow_corr_gate

    hybrid_output = output_v3.copy()
    hybrid_output[selected] = output_v21[selected]
    final_slow_corr, final_delta_relative, final_correlation = slow_corr_mask(
        hybrid_output,
        sample_rate,
        args.slow_corr_delta_rel_threshold,
        args.slow_corr_correlation_threshold,
    )
    final_new_slow_corr = final_slow_corr & ~v3_slow_corr
    if final_new_slow_corr.any():
        bad_windows = np.flatnonzero(final_new_slow_corr)[:10].tolist()
        raise AssertionError(
            "v2.5 SLOW_CORR gate failed for window rows: "
            + ", ".join(str(value + 1) for value in bad_windows)
        )

    payload["denoised_eeg"] = hybrid_output.astype(np.float32)
    difference = hybrid_output - source
    payload["input_to_output_rmse"] = np.sqrt(
        np.mean(np.square(difference), axis=2)
    ).astype(np.float32)
    low_correlation = np.asarray(
        payload["hybrid_v21_low_drift_highpass_correlation_eeg"],
        dtype=np.float64,
    )
    low_rrmse = np.asarray(
        payload["hybrid_v21_low_drift_highpass_rrmse_eeg"],
        dtype=np.float64,
    )
    _restore_selective_masks(
        payload,
        v3,
        selected,
        low_correlation,
        low_rrmse,
    )

    policy_json = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    payload.update(
        {
            "hybrid_policy_version": np.array(HYBRID_POLICY_VERSION),
            "hybrid_policy_json": np.array(policy_json),
            "hybrid_v21_v24_reference_selected_eeg": v24_selected.astype(
                np.int8
            ),
            "hybrid_v21_selected_eeg": selected.astype(np.int8),
            "hybrid_v21_absolute_power_ratio_4_30_eeg": power_ratio.astype(
                np.float32
            ),
            "hybrid_v21_absolute_power_4_30_pass_eeg": power_gate.astype(
                np.int8
            ),
            "hybrid_v3_slow_corr_window": v3_slow_corr.astype(np.int8),
            "hybrid_v21_tentative_slow_corr_window": (
                tentative_slow_corr.astype(np.int8)
            ),
            "hybrid_v21_new_slow_corr_window": new_slow_corr.astype(np.int8),
            "hybrid_v21_slow_corr_gate_pass_eeg": slow_corr_gate.astype(
                np.int8
            ),
            "hybrid_final_slow_corr_window": final_slow_corr.astype(np.int8),
            "hybrid_v3_slow_corr_delta_rel_avg": v3_delta_relative.astype(
                np.float32
            ),
            "hybrid_v3_slow_corr_channel_correlation": (
                v3_channel_correlation.astype(np.float32)
            ),
            "hybrid_v21_tentative_slow_corr_delta_rel_avg": (
                tentative_delta_relative.astype(np.float32)
            ),
            "hybrid_v21_tentative_slow_corr_channel_correlation": (
                tentative_correlation.astype(np.float32)
            ),
            "hybrid_final_slow_corr_delta_rel_avg": (
                final_delta_relative.astype(np.float32)
            ),
            "hybrid_final_slow_corr_channel_correlation": (
                final_correlation.astype(np.float32)
            ),
            "hybrid_v21_selected_count": np.array(
                int(selected.sum()), dtype=np.int64
            ),
        }
    )

    row_lookup = {
        (int(row["artifact_row_number"]) - 1, int(str(row["channel"])[-1]) - 1): row
        for row in rows
    }
    candidate = np.asarray(payload["hybrid_v21_candidate_eeg"], dtype=bool)
    for window_index, channel_index in np.argwhere(candidate):
        row = row_lookup[(int(window_index), int(channel_index))]
        was_v24_selected = bool(v24_selected[window_index, channel_index])
        passed_power = bool(power_gate[window_index, channel_index])
        passed_slow_corr = bool(slow_corr_gate[window_index, channel_index])
        is_selected = bool(selected[window_index, channel_index])
        reasons = [
            value
            for value in str(row["rejection_reasons"]).split("|")
            if value
        ]
        if was_v24_selected and not passed_power:
            reasons.append("absolute_power_4_30_failed")
        if (
            was_v24_selected
            and passed_power
            and not passed_slow_corr
        ):
            reasons.append("new_slow_corr")
        if is_selected and reasons:
            raise AssertionError("Selected v2.5 row contains rejection reasons.")
        row.update(
            {
                "decision": "use_v21" if is_selected else "keep_v3select",
                "selected_v21": is_selected,
                "selected_v24_reference": was_v24_selected,
                "rejection_reasons": "|".join(reasons),
                "v21_absolute_power_ratio_4_30": float(
                    power_ratio[window_index, channel_index]
                ),
                "absolute_power_4_30_pass": passed_power,
                "v3_slow_corr": bool(v3_slow_corr[window_index]),
                "tentative_slow_corr": bool(
                    tentative_slow_corr[window_index]
                ),
                "new_slow_corr": bool(new_slow_corr[window_index]),
                "slow_corr_gate_pass": passed_slow_corr,
            }
        )

    counts.update(
        {
            "v24_reference_selected": int(v24_selected.sum()),
            "absolute_power_4_30_rejected_from_v24": int(
                (v24_selected & ~power_gate).sum()
            ),
            "new_slow_corr_windows": int(new_slow_corr.sum()),
            "new_slow_corr_rejected_from_power_pass": int(
                (power_selected & ~slow_corr_gate).sum()
            ),
            "hybrid_v21_selected": int(selected.sum()),
            "hybrid_v21_rejected": int(candidate.sum() - selected.sum()),
        }
    )
    return payload, rows, counts


def validate_v25_payload(
    payload: dict[str, np.ndarray],
    expected_policy: dict[str, Any],
    path: Path,
) -> None:
    _validate_base_payload(payload, expected_policy, path)
    required = {
        "hybrid_v21_v24_reference_selected_eeg",
        "hybrid_v21_absolute_power_ratio_4_30_eeg",
        "hybrid_v21_absolute_power_4_30_pass_eeg",
        "hybrid_v21_new_slow_corr_window",
        "hybrid_v21_slow_corr_gate_pass_eeg",
        "hybrid_final_slow_corr_window",
        "hybrid_v3_slow_corr_window",
    }
    require_keys(payload, required, path)
    selected = np.asarray(payload["hybrid_v21_selected_eeg"], dtype=bool)
    v24_selected = np.asarray(
        payload["hybrid_v21_v24_reference_selected_eeg"], dtype=bool
    )
    power_pass = np.asarray(
        payload["hybrid_v21_absolute_power_4_30_pass_eeg"], dtype=bool
    )
    slow_pass = np.asarray(
        payload["hybrid_v21_slow_corr_gate_pass_eeg"], dtype=bool
    )
    if not np.all(~selected | (v24_selected & power_pass & slow_pass)):
        raise ValueError(f"{path.name} contains a selection outside v2.5 gates.")
    final_slow = np.asarray(payload["hybrid_final_slow_corr_window"], dtype=bool)
    base_slow = np.asarray(payload["hybrid_v3_slow_corr_window"], dtype=bool)
    if np.any(final_slow & ~base_slow):
        raise ValueError(f"{path.name} introduces a new SLOW_CORR window.")


@dataclass(frozen=True)
class HybridConfig:
    """Application settings; model weights and numerical gates stay separate."""

    v2_checkpoint: Path
    v3_checkpoint: Path
    device_name: str = "auto"
    batch_size: int = 256
    ecg_min_abs_correlation: float = 0.05
    preservation_correlation_min: float = 0.90
    preservation_rrmse_max: float = 0.50
    low_drift_highpass_hz: float = 1.0
    edge_trim_sec: float = 0.25
    absolute_power_4_30_min: float = 0.80
    slow_corr_delta_rel_threshold: float = 0.60
    slow_corr_correlation_threshold: float = 0.30

    def arguments(self) -> argparse.Namespace:
        values = asdict(self)
        for key in ("v2_checkpoint", "v3_checkpoint", "device_name", "batch_size"):
            values.pop(key)
        for name, value in values.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"Hybrid parameter {name} must be finite.")
        args = argparse.Namespace(**values)
        validate_numeric_args(args)
        if self.batch_size < 1:
            raise ValueError("Hybrid batch_size must be positive.")
        return args


class HybridDenoiser:
    """Load both fixed checkpoints once and apply the complete v2.5 module.

    ``assess_v21`` evaluates the generated v2 candidate through ADE using the
    current recording's original, full-precision thresholds. It returns the
    freshly written artifact workbook. Historical results are never required.
    """

    def __init__(self, config: HybridConfig):
        self.config = config
        self.args = config.arguments()
        self.policy = hybrid_policy(self.args)
        self._models = None

    def load_models(self):
        if self._models is None:
            from .checkpoints import load_model
            from .runtime import resolve_device

            device = resolve_device(self.config.device_name)
            print("Hybrid inference device:", device, flush=True)
            models = []
            for checkpoint_path in (
                self.config.v2_checkpoint, self.config.v3_checkpoint
            ):
                print("Loading checkpoint:", checkpoint_path, flush=True)
                model, checkpoint = load_model(checkpoint_path, device=device)
                if checkpoint["model_config"]["variant"] != "conditioned":
                    raise ValueError("The v2.5 hybrid requires conditioned checkpoints.")
                models.append((model, checkpoint, Path(checkpoint_path)))
            self._models = (device, models)
        return self._models

    def apply_recording(
        self, *, recording_id: str, input_path: Path,
        original_artifact_path: Path, work_dir: Path, output_path: Path,
        assess_v21: Callable[[Path, Path], Path],
    ) -> Path:
        from .inference import InferencePolicy, apply_loaded_model_to_recording
        from .io import load_dose_recording_npz

        record = load_dose_recording_npz(input_path)
        for key, expected in (
            ("sample_rate", 125.0), ("window_seconds", 5.0),
            ("window_overlap", 0.0),
        ):
            actual = float(np.asarray(record[key]).item())
            if not np.isclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError(
                    f"The supplied hybrid is validated for {key}={expected:g}; "
                    f"received {actual:g}. Use the original signal or a matching model."
                )
        if scalar_text(record, "signal_method") != "original":
            raise ValueError("Hybrid input must be the original signal variant.")
        if scalar_text(record, "recording_id") != recording_id:
            raise ValueError("Hybrid input belongs to another recording.")
        device, models = self.load_models()
        v21_dir, v3_dir = Path(work_dir) / "v21", Path(work_dir) / "v3"
        v21_path = v21_dir / f"{recording_id}_windows_denoised.npz"
        v3_path = v3_dir / f"{recording_id}_windows_denoised.npz"
        policies = (
            InferencePolicy(
                version="v2.1", output_safety_fallback=False,
                selective_preservation_fallback=False,
                ecg_min_abs_correlation=self.config.ecg_min_abs_correlation,
            ),
            InferencePolicy(
                ecg_min_abs_correlation=self.config.ecg_min_abs_correlation,
                preservation_correlation_min=self.config.preservation_correlation_min,
                preservation_rrmse_max=self.config.preservation_rrmse_max,
                low_drift_highpass_hz=self.config.low_drift_highpass_hz,
                edge_trim_sec=self.config.edge_trim_sec,
            ),
        )
        # Always regenerate for this input/configuration. A same-name result
        # from an earlier run is never silently treated as a valid cache.
        for (model, checkpoint, checkpoint_path), policy, candidate_path in zip(
            models, policies, (v21_path, v3_path)
        ):
            print(f"Hybrid candidate {policy.version}: {recording_id}", flush=True)
            apply_loaded_model_to_recording(
                model=model, checkpoint=checkpoint,
                checkpoint_path=checkpoint_path, input_path=input_path,
                output_path=candidate_path, policy=policy,
                batch_size=self.config.batch_size, device=device,
            )
        v21_artifact_path = Path(assess_v21(v21_dir, Path(work_dir) / "assessment"))
        payload, rows, counts = build_hybrid(
            recording_id=recording_id, v21_path=v21_path, v3_path=v3_path,
            original_artifact_path=Path(original_artifact_path),
            v21_artifact_path=v21_artifact_path,
            args=self.args, policy=self.policy,
        )
        validate_v25_payload(payload, self.policy, Path(output_path))
        atomic_save_npz(Path(output_path), payload)
        # Decisions are part of application provenance, not development data.
        pd.DataFrame(rows).to_csv(Path(work_dir) / "hybrid_decisions.csv", index=False)
        (Path(work_dir) / "hybrid_summary.json").write_text(
            json.dumps({"recording_id": recording_id, "policy": self.policy,
                        "counts": counts}, indent=2), encoding="utf-8"
        )
        print("Hybrid selected v2 LOW_DRIFT candidates:",
              counts["hybrid_v21_selected"], flush=True)
        return Path(output_path)
