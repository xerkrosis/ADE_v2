"""Stable on-disk format for DOSE-I autoencoder window exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np


SCHEMA_VERSION = 1

# Exported for each EEG channel.  The order is written into every NPZ file and
# is mapped to the checkpoint marker order by the PyTorch Dataset.
PIPELINE_MARKERS = (
    "SLOW_CORR",
    "EMG",
    "ECG",
    "BAD_AUTO",
    "WAVELET_HF",
    "WAVELET_TRANSIENT",
    "AMP_BURST",
    "JUMP",
    "FLATLINE",
    "HF_EMG",
    "LOW_DRIFT",
    "CLIPPING",
)

CHANNEL_MARKER_COLUMNS = {
    "SLOW_CORR": ("has_slow_corr", "has_slow_corr"),
    "EMG": ("has_emg_eeg1", "has_emg_eeg2"),
    "ECG": ("has_ecg_eeg1", "has_ecg_eeg2"),
    "BAD_AUTO": ("has_bad_auto_eeg1", "has_bad_auto_eeg2"),
    "WAVELET_HF": ("has_wavelet_hf_eeg1", "has_wavelet_hf_eeg2"),
    "WAVELET_TRANSIENT": (
        "has_wavelet_transient_eeg1",
        "has_wavelet_transient_eeg2",
    ),
    "AMP_BURST": ("has_amp_burst_eeg1", "has_amp_burst_eeg2"),
    "JUMP": ("has_jump_eeg1", "has_jump_eeg2"),
    "FLATLINE": ("has_flatline_eeg1", "has_flatline_eeg2"),
    "HF_EMG": ("has_hf_emg_eeg1", "has_hf_emg_eeg2"),
    "LOW_DRIFT": ("has_low_drift_eeg1", "has_low_drift_eeg2"),
    "CLIPPING": ("has_clipping_eeg1", "has_clipping_eeg2"),
}

LABEL_COLUMNS = {
    "quality": (
        "quality_label_eeg1",
        "quality_label_eeg2",
        "quality_label",
    ),
    "usable": (
        "usable_label_eeg1",
        "usable_label_eeg2",
        "usable_label",
    ),
    "strict_clean": (
        "strict_clean_label_eeg1",
        "strict_clean_label_eeg2",
        "strict_clean_label",
    ),
}


def _as_window_matrix(windows: Any, name: str) -> np.ndarray:
    array = np.asarray(windows, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must have shape [windows, samples], got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _column(df_features, name: str, window_count: int) -> np.ndarray:
    if name not in df_features.columns:
        raise KeyError(f"Missing pipeline column for autoencoder export: {name}")
    values = df_features[name].astype(np.int8).to_numpy()
    if values.shape != (window_count,):
        raise ValueError(
            f"Column {name} has shape {values.shape}; expected {(window_count,)}."
        )
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"Column {name} contains values other than 0 and 1.")
    return values


def export_dose_recording_npz(
    *,
    output_dir: str | Path,
    recording_id: str,
    eeg1_windows,
    eeg2_windows,
    ecg_windows,
    df_features,
    sample_rate: float,
    window_seconds: float,
    window_overlap: float,
    begin_ns,
    end_ns,
    signal_method: str,
    preprocessing: dict,
) -> Path:
    """Write one recording without requiring PyTorch in the main pipeline."""

    eeg1 = _as_window_matrix(eeg1_windows, "eeg1_windows")
    eeg2 = _as_window_matrix(eeg2_windows, "eeg2_windows")
    if eeg1.shape != eeg2.shape:
        raise ValueError(
            f"EEG window shapes differ: EEG1={eeg1.shape}, EEG2={eeg2.shape}."
        )

    window_count, sample_count = eeg1.shape
    if len(df_features) != window_count:
        raise ValueError(
            "Feature/window count mismatch during autoencoder export: "
            f"features={len(df_features)}, signals={window_count}."
        )

    eeg = np.stack([eeg1, eeg2], axis=1).astype(np.float32, copy=False)

    if ecg_windows is None:
        ecg = np.zeros((window_count, 1, sample_count), dtype=np.float32)
        ecg_available = np.array(False, dtype=np.bool_)
    else:
        ecg_matrix = _as_window_matrix(ecg_windows, "ecg_windows")
        if ecg_matrix.shape != eeg1.shape:
            raise ValueError(
                f"ECG window shape {ecg_matrix.shape} does not match "
                f"EEG shape {eeg1.shape}."
            )
        ecg = ecg_matrix[:, None, :]
        ecg_available = np.array(True, dtype=np.bool_)

    markers = np.empty(
        (window_count, 2, len(PIPELINE_MARKERS)),
        dtype=np.int8,
    )
    for marker_index, marker_name in enumerate(PIPELINE_MARKERS):
        eeg1_column, eeg2_column = CHANNEL_MARKER_COLUMNS[marker_name]
        markers[:, 0, marker_index] = _column(
            df_features, eeg1_column, window_count
        )
        markers[:, 1, marker_index] = _column(
            df_features, eeg2_column, window_count
        )

    payload: dict[str, np.ndarray] = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int16),
        "recording_id": np.array(str(recording_id)),
        "signal_method": np.array(str(signal_method)),
        "sample_rate": np.array(float(sample_rate), dtype=np.float64),
        "window_seconds": np.array(float(window_seconds), dtype=np.float64),
        "window_overlap": np.array(float(window_overlap), dtype=np.float64),
        "window_number": np.arange(window_count, dtype=np.int64),
        "begin_ns": np.asarray(begin_ns, dtype=np.int64),
        "end_ns": np.asarray(end_ns, dtype=np.int64),
        "eeg": eeg,
        "ecg": ecg,
        "ecg_available": ecg_available,
        "marker_names": np.asarray(PIPELINE_MARKERS),
        "markers": markers,
        "preprocessing_json": np.array(
            json.dumps(preprocessing, sort_keys=True, default=str)
        ),
    }

    for label_name, (eeg1_column, eeg2_column, combined_column) in (
        LABEL_COLUMNS.items()
    ):
        payload[f"{label_name}_eeg"] = np.stack(
            [
                _column(df_features, eeg1_column, window_count),
                _column(df_features, eeg2_column, window_count),
            ],
            axis=1,
        )
        payload[f"{label_name}_combined"] = _column(
            df_features, combined_column, window_count
        )

    if np.any(payload["strict_clean_eeg"] > payload["usable_eeg"]):
        raise ValueError(
            "Channel-specific strict_clean labels must be a subset of usable."
        )
    if np.any(payload["usable_eeg"] > payload["quality_eeg"]):
        raise ValueError(
            "Channel-specific usable labels must be a subset of quality."
        )
    if np.any(
        payload["strict_clean_combined"] > payload["usable_combined"]
    ):
        raise ValueError("Combined strict_clean must be a subset of usable.")

    for key in ("begin_ns", "end_ns"):
        if payload[key].shape != (window_count,):
            raise ValueError(
                f"{key} has shape {payload[key].shape}; expected "
                f"{(window_count,)}."
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{recording_id}_windows.npz"
    temporary_path = output_path.with_suffix(".npz.tmp")

    with temporary_path.open("wb") as file_handle:
        np.savez_compressed(file_handle, **payload)

    temporary_path.replace(output_path)
    return output_path


def load_dose_recording_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load and minimally validate one exported recording."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}

    schema_version = int(result.get("schema_version", -1))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported autoencoder export schema {schema_version} in {path}; "
            f"expected {SCHEMA_VERSION}."
        )

    required = {
        "recording_id",
        "sample_rate",
        "window_number",
        "begin_ns",
        "end_ns",
        "eeg",
        "ecg",
        "ecg_available",
        "marker_names",
        "markers",
        "strict_clean_eeg",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise KeyError(f"{path} is missing NPZ fields: {', '.join(missing)}")

    eeg = result.get("eeg")
    if eeg is None or eeg.ndim != 3 or eeg.shape[1] != 2:
        raise ValueError(
            f"Invalid EEG tensor in {path}; expected [windows, 2, samples]."
        )
    if not np.isfinite(eeg).all():
        raise ValueError(f"EEG tensor in {path} contains non-finite values.")
    window_count, _, sample_count = eeg.shape
    if result["ecg"].shape != (window_count, 1, sample_count):
        raise ValueError(f"ECG tensor has an incompatible shape in {path}.")
    if result["markers"].shape[:2] != (window_count, 2):
        raise ValueError(f"Marker tensor has an incompatible shape in {path}.")
    if result["markers"].shape[2] != len(result["marker_names"]):
        raise ValueError(f"Marker names/tensor disagree in {path}.")
    for key in ("window_number", "begin_ns", "end_ns"):
        if result[key].shape != (window_count,):
            raise ValueError(f"{key} has an incompatible shape in {path}.")
    if float(np.asarray(result["sample_rate"]).item()) <= 0:
        raise ValueError(f"sample_rate must be positive in {path}.")
    return result
