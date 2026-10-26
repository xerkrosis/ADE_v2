"""Apply trained denoising models to real exported DOSE-I windows.

Model version and inference policy are deliberately separate.  The v2.3 policy
keeps the trained checkpoint unchanged while making its real-data use more
selective:

* strict-clean channel windows are copied exactly;
* irrecoverable FLATLINE and CLIPPING windows are copied exactly;
* an ECG reference is supplied only when its absolute linear coupling to the
  respective EEG channel reaches a configurable minimum;
* model outputs that newly introduce FLATLINE or CLIPPING are replaced by the
  corresponding unchanged input channel window;
* nearly signal-free inputs are copied without model application;
* LOW_DRIFT outputs must preserve the high-passed waveform;
* AMP_BURST/JUMP outputs must preserve the signal outside the localized event;
* ECG outputs must not increase absolute EEG-ECG coupling;
* all policy decisions are stored channel-wise in the prediction NPZ.

The preservation thresholds are deliberately marked as exploratory policy
parameters.  They protect the current real-data experiment; they are not
claimed to be clinically validated acceptance limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .io import load_dose_recording_npz
from .markers import derive_recording_thresholds, signal_features


EPSILON = 1e-8
CORRELATION_EPSILON = 1e-12
INFERENCE_POLICY_VERSION = "v2.3"
OUTPUT_SAFETY_POLICY_VERSION = "v2.2"
LEGACY_INFERENCE_POLICY_VERSION = "v2.1"
SUPPORTED_INFERENCE_POLICY_VERSIONS = (
    LEGACY_INFERENCE_POLICY_VERSION,
    OUTPUT_SAFETY_POLICY_VERSION,
    INFERENCE_POLICY_VERSION,
)
DEFAULT_BYPASS_MARKERS = ("FLATLINE", "CLIPPING")


@dataclass(frozen=True)
class InferencePolicy:
    """Conservative rules applied around an unchanged trained checkpoint."""

    version: str = INFERENCE_POLICY_VERSION
    preserve_strict_clean: bool = True
    bypass_markers: tuple[str, ...] = DEFAULT_BYPASS_MARKERS
    ecg_min_abs_correlation: float = 0.0
    output_safety_fallback: bool = True
    selective_preservation_fallback: bool = True
    low_information_relative_floor: float = 1e-4
    preservation_correlation_min: float = 0.90
    preservation_rrmse_max: float = 0.50
    low_drift_highpass_hz: float = 1.0
    edge_trim_sec: float = 0.25
    event_z_threshold: float = 6.0
    event_padding_sec: float = 0.20
    ecg_nonworsening_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.version not in SUPPORTED_INFERENCE_POLICY_VERSIONS:
            raise ValueError(
                f"Unsupported inference policy {self.version!r}; "
                f"expected one of {SUPPORTED_INFERENCE_POLICY_VERSIONS!r}."
            )
        if self.version == LEGACY_INFERENCE_POLICY_VERSION and (
            self.output_safety_fallback
            or self.selective_preservation_fallback
        ):
            raise ValueError(
                "The v2.1 policy includes neither safety fallback. Set both "
                "fallback fields to False or use a newer policy."
            )
        if (
            self.version == OUTPUT_SAFETY_POLICY_VERSION
            and self.selective_preservation_fallback
        ):
            raise ValueError(
                "The v2.2 policy does not include the selective preservation "
                "fallback. Use version='v2.3' or set "
                "selective_preservation_fallback=False."
            )
        if self.version == INFERENCE_POLICY_VERSION and (
            not self.output_safety_fallback
            or not self.selective_preservation_fallback
        ):
            raise ValueError(
                "Inference policy v2.3 requires both output_safety_fallback "
                "and selective_preservation_fallback to be enabled."
            )
        threshold = float(self.ecg_min_abs_correlation)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("ecg_min_abs_correlation must be within [0, 1].")
        markers = tuple(str(value).upper() for value in self.bypass_markers)
        if len(set(markers)) != len(markers):
            raise ValueError("bypass_markers must not contain duplicates.")
        object.__setattr__(self, "bypass_markers", markers)
        object.__setattr__(self, "ecg_min_abs_correlation", threshold)

        bounded = {
            "low_information_relative_floor": (
                self.low_information_relative_floor,
                0.0,
                1.0,
                False,
            ),
            "preservation_correlation_min": (
                self.preservation_correlation_min,
                -1.0,
                1.0,
                True,
            ),
            "preservation_rrmse_max": (
                self.preservation_rrmse_max,
                0.0,
                np.inf,
                True,
            ),
            "low_drift_highpass_hz": (
                self.low_drift_highpass_hz,
                0.0,
                np.inf,
                False,
            ),
            "edge_trim_sec": (self.edge_trim_sec, 0.0, np.inf, True),
            "event_z_threshold": (
                self.event_z_threshold,
                0.0,
                np.inf,
                False,
            ),
            "event_padding_sec": (
                self.event_padding_sec,
                0.0,
                np.inf,
                True,
            ),
            "ecg_nonworsening_tolerance": (
                self.ecg_nonworsening_tolerance,
                0.0,
                1.0,
                True,
            ),
        }
        for name, (raw_value, minimum, maximum, minimum_inclusive) in bounded.items():
            value = float(raw_value)
            lower_ok = value >= minimum if minimum_inclusive else value > minimum
            if not np.isfinite(value) or not lower_ok or value > maximum:
                lower = "[" if minimum_inclusive else "("
                raise ValueError(
                    f"{name} must be within {lower}{minimum}, {maximum}]."
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bypass_markers"] = list(self.bypass_markers)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "InferencePolicy":
        values = dict(values)
        if "bypass_markers" in values:
            values["bypass_markers"] = tuple(values["bypass_markers"])
        if values.get("version") == LEGACY_INFERENCE_POLICY_VERSION:
            values.setdefault("output_safety_fallback", False)
            values.setdefault("selective_preservation_fallback", False)
        if values.get("version") == OUTPUT_SAFETY_POLICY_VERSION:
            values.setdefault("selective_preservation_fallback", False)
        return cls(**values)

    @property
    def tag(self) -> str:
        threshold = f"{self.ecg_min_abs_correlation:.3f}".rstrip("0").rstrip(".")
        threshold = threshold.replace(".", "p") or "0"
        version = self.version.removeprefix("v").replace(".", "p")
        if self.selective_preservation_fallback:
            safety = "safe-selective"
        else:
            safety = "safe" if self.output_safety_fallback else "raw"
        return f"policy-v{version}_ecgmin-{threshold}_{safety}"


def policy_json(policy: InferencePolicy) -> str:
    return json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"))


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(len(left), np.nan, dtype=np.float64)
    valid = denominator > CORRELATION_EPSILON
    result[valid] = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return result


def _row_rrmse(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    difference_rms = np.sqrt(np.mean(np.square(candidate - reference), axis=1))
    centered = reference - reference.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(np.square(centered), axis=1))
    result = np.full(len(reference), np.nan, dtype=np.float64)
    valid = scale > CORRELATION_EPSILON
    result[valid] = difference_rms[valid] / scale[valid]
    return result


def _marker_mask(
    record: dict[str, np.ndarray],
    marker_name: str,
) -> np.ndarray:
    exported_names = [str(value) for value in record["marker_names"].tolist()]
    exported = np.asarray(record["markers"], dtype=bool)
    if exported.ndim != 3 or exported.shape[1] != 2:
        raise ValueError(
            "markers must have shape [windows, 2, marker_names], "
            f"got {exported.shape}."
        )
    if exported.shape[2] != len(exported_names):
        raise ValueError("markers and marker_names have incompatible shapes.")
    if marker_name not in exported_names:
        return np.zeros(exported.shape[:2], dtype=bool)
    return exported[:, :, exported_names.index(marker_name)]


def detect_low_information_inputs(
    input_eeg: np.ndarray,
    relative_floor: float,
) -> dict[str, np.ndarray]:
    """Identify nearly signal-free channels relative to their recording median."""

    input_eeg = np.asarray(input_eeg, dtype=np.float64)
    if input_eeg.ndim != 3 or input_eeg.shape[1] != 2:
        raise ValueError(
            "input_eeg must have shape [windows, 2, samples], "
            f"got {input_eeg.shape}."
        )
    centered = input_eeg - input_eeg.mean(axis=2, keepdims=True)
    centered_rms = np.sqrt(np.mean(np.square(centered), axis=2))
    peak_to_peak = np.ptp(input_eeg, axis=2)
    rms_floor = np.median(centered_rms, axis=0) * float(relative_floor)
    ptp_floor = np.median(peak_to_peak, axis=0) * float(relative_floor)
    candidate = (
        (centered_rms <= rms_floor[None, :])
        & (peak_to_peak <= ptp_floor[None, :])
    )
    return {
        "candidate_eeg": candidate,
        "centered_rms_eeg": centered_rms.astype(np.float32),
        "peak_to_peak_eeg": peak_to_peak.astype(np.float32),
        "rms_floor_by_channel": rms_floor.astype(np.float32),
        "peak_to_peak_floor_by_channel": ptp_floor.astype(np.float32),
    }


def _filter_rows_highpass(
    values: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(
            f"Invalid high-pass cutoff {cutoff_hz:g} Hz for fs={sample_rate:g}."
        )
    sections = butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(
        sections,
        np.asarray(values, dtype=np.float64),
        axis=1,
    )


def _trim_rows(values: np.ndarray, trim_samples: int) -> np.ndarray:
    if trim_samples <= 0:
        return values
    if values.shape[1] <= 2 * trim_samples + 8:
        raise ValueError(
            f"Edge trim {trim_samples} leaves too few of "
            f"{values.shape[1]} samples."
        )
    return values[:, trim_samples:-trim_samples]


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= CORRELATION_EPSILON:
        scale = float(np.std(values))
    if scale <= CORRELATION_EPSILON:
        return np.zeros_like(values)
    return np.abs(values - median) / scale


def _dilate_mask(mask: np.ndarray, padding: int) -> np.ndarray:
    if padding <= 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones(2 * padding + 1, dtype=np.int16)
    return np.convolve(np.asarray(mask, dtype=np.int16), kernel, mode="same") > 0


def _localize_event(
    signal: np.ndarray,
    *,
    amp_burst: bool,
    jump: bool,
    z_threshold: float,
    padding_samples: int,
) -> np.ndarray:
    mask = np.zeros(len(signal), dtype=bool)
    if amp_burst:
        amplitude_z = _robust_z(signal)
        amplitude_mask = amplitude_z >= z_threshold
        if not amplitude_mask.any():
            amplitude_mask[int(np.argmax(amplitude_z))] = True
        mask |= amplitude_mask
    if jump:
        difference = np.diff(signal, prepend=signal[0])
        difference_z = _robust_z(difference)
        jump_mask = difference_z >= z_threshold
        if not jump_mask.any():
            jump_mask[int(np.argmax(difference_z))] = True
        mask |= jump_mask
    return _dilate_mask(mask, padding_samples)


def apply_selective_preservation_fallback(
    *,
    record: dict[str, np.ndarray],
    input_eeg: np.ndarray,
    candidate_eeg: np.ndarray,
    process_eeg: np.ndarray,
    output_safety_fallback_eeg: np.ndarray,
    ecg_reference_active_eeg: np.ndarray,
    sample_rate: float,
    policy: InferencePolicy,
) -> dict[str, Any]:
    """Apply marker-specific preservation gates to otherwise accepted outputs.

    Only processed channel windows that survived the existing v2.2
    FLATLINE/CLIPPING gate are eligible. A failed marker-specific check copies
    the corresponding unchanged input channel window exactly.
    """

    input_eeg = np.asarray(input_eeg, dtype=np.float32)
    candidate_eeg = np.asarray(candidate_eeg, dtype=np.float32)
    process_eeg = np.asarray(process_eeg, dtype=bool)
    prior_fallback = np.asarray(output_safety_fallback_eeg, dtype=bool)
    ecg_active = np.asarray(ecg_reference_active_eeg, dtype=bool)
    if input_eeg.ndim != 3 or input_eeg.shape[1] != 2:
        raise ValueError(
            "input_eeg must have shape [windows, 2, samples], "
            f"got {input_eeg.shape}."
        )
    if candidate_eeg.shape != input_eeg.shape:
        raise ValueError("candidate_eeg must match input_eeg.")
    decision_shape = input_eeg.shape[:2]
    for name, value in (
        ("process_eeg", process_eeg),
        ("output_safety_fallback_eeg", prior_fallback),
        ("ecg_reference_active_eeg", ecg_active),
    ):
        if value.shape != decision_shape:
            raise ValueError(
                f"{name} must have shape {decision_shape}, got {value.shape}."
            )
    sample_rate = float(sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be a positive finite value.")

    eligible = process_eeg & ~prior_fallback
    low_drift_marker = _marker_mask(record, "LOW_DRIFT")
    amp_burst_marker = _marker_mask(record, "AMP_BURST")
    jump_marker = _marker_mask(record, "JUMP")
    ecg_marker = _marker_mask(record, "ECG")

    low_drift_checked = eligible & low_drift_marker
    event_checked = eligible & (amp_burst_marker | jump_marker)
    ecg_checked = eligible & ecg_marker & ecg_active
    checked = low_drift_checked | event_checked | ecg_checked

    low_drift_correlation = np.full(decision_shape, np.nan, dtype=np.float32)
    low_drift_rrmse = np.full(decision_shape, np.nan, dtype=np.float32)
    event_outside_correlation = np.full(decision_shape, np.nan, dtype=np.float32)
    event_outside_rrmse = np.full(decision_shape, np.nan, dtype=np.float32)
    ecg_input_abs_correlation = np.full(decision_shape, np.nan, dtype=np.float32)
    ecg_output_abs_correlation = np.full(decision_shape, np.nan, dtype=np.float32)
    low_drift_failed = np.zeros(decision_shape, dtype=bool)
    event_failed = np.zeros(decision_shape, dtype=bool)
    ecg_failed = np.zeros(decision_shape, dtype=bool)

    samples = input_eeg.shape[-1]
    flattened_input = input_eeg.reshape(-1, samples)
    flattened_candidate = candidate_eeg.reshape(-1, samples)
    trim_samples = int(round(policy.edge_trim_sec * sample_rate))

    low_drift_indices = np.flatnonzero(low_drift_checked.reshape(-1))
    if len(low_drift_indices):
        filtered_input = _filter_rows_highpass(
            flattened_input[low_drift_indices],
            sample_rate,
            policy.low_drift_highpass_hz,
        )
        filtered_candidate = _filter_rows_highpass(
            flattened_candidate[low_drift_indices],
            sample_rate,
            policy.low_drift_highpass_hz,
        )
        filtered_input = _trim_rows(filtered_input, trim_samples)
        filtered_candidate = _trim_rows(filtered_candidate, trim_samples)
        correlations = _row_correlation(filtered_input, filtered_candidate)
        rrmse = _row_rrmse(filtered_input, filtered_candidate)
        local_failed = (
            ~np.isfinite(correlations)
            | ~np.isfinite(rrmse)
            | (correlations < policy.preservation_correlation_min)
            | (rrmse > policy.preservation_rrmse_max)
        )
        low_drift_correlation.reshape(-1)[low_drift_indices] = correlations
        low_drift_rrmse.reshape(-1)[low_drift_indices] = rrmse
        low_drift_failed.reshape(-1)[low_drift_indices] = local_failed

    padding_samples = int(round(policy.event_padding_sec * sample_rate))
    event_indices = np.flatnonzero(event_checked.reshape(-1))
    flattened_amp = amp_burst_marker.reshape(-1)
    flattened_jump = jump_marker.reshape(-1)
    for flattened_index in event_indices:
        event_mask = _localize_event(
            flattened_input[flattened_index],
            amp_burst=bool(flattened_amp[flattened_index]),
            jump=bool(flattened_jump[flattened_index]),
            z_threshold=policy.event_z_threshold,
            padding_samples=padding_samples,
        )
        keep = ~event_mask
        if trim_samples > 0:
            keep[:trim_samples] = False
            keep[-trim_samples:] = False
        if int(keep.sum()) < max(16, int(0.20 * samples)):
            correlation = float("nan")
            rrmse = float("nan")
        else:
            reference = flattened_input[flattened_index, keep][None, :]
            candidate = flattened_candidate[flattened_index, keep][None, :]
            correlation = float(_row_correlation(reference, candidate)[0])
            rrmse = float(_row_rrmse(reference, candidate)[0])
        row, channel = np.unravel_index(flattened_index, decision_shape)
        event_outside_correlation[row, channel] = correlation
        event_outside_rrmse[row, channel] = rrmse
        event_failed[row, channel] = (
            not np.isfinite(correlation)
            or not np.isfinite(rrmse)
            or correlation < policy.preservation_correlation_min
            or rrmse > policy.preservation_rrmse_max
        )

    ecg_available = bool(np.asarray(record["ecg_available"]).item())
    if np.any(ecg_checked) and not ecg_available:
        raise ValueError("ECG gate is active although no ECG reference is available.")
    if ecg_available:
        ecg = np.asarray(record["ecg"], dtype=np.float32)
        if ecg.shape != (input_eeg.shape[0], 1, samples):
            raise ValueError(
                "ECG must have shape "
                f"{(input_eeg.shape[0], 1, samples)}, got {ecg.shape}."
            )
        repeated_ecg = np.repeat(ecg, input_eeg.shape[1], axis=1).reshape(
            -1, samples
        )
        ecg_indices = np.flatnonzero(ecg_checked.reshape(-1))
        if len(ecg_indices):
            input_correlation = np.abs(
                _row_correlation(
                    flattened_input[ecg_indices],
                    repeated_ecg[ecg_indices],
                )
            )
            output_correlation = np.abs(
                _row_correlation(
                    flattened_candidate[ecg_indices],
                    repeated_ecg[ecg_indices],
                )
            )
            local_failed = (
                ~np.isfinite(input_correlation)
                | ~np.isfinite(output_correlation)
                | (
                    output_correlation
                    > input_correlation + policy.ecg_nonworsening_tolerance
                )
            )
            ecg_input_abs_correlation.reshape(-1)[ecg_indices] = input_correlation
            ecg_output_abs_correlation.reshape(-1)[ecg_indices] = output_correlation
            ecg_failed.reshape(-1)[ecg_indices] = local_failed

    fallback = low_drift_failed | event_failed | ecg_failed
    safe_eeg = candidate_eeg.copy()
    safe_eeg[fallback] = input_eeg[fallback]
    return {
        "denoised_eeg": safe_eeg,
        "checked_eeg": checked,
        "fallback_eeg": fallback,
        "low_drift_checked_eeg": low_drift_checked,
        "low_drift_failed_eeg": low_drift_failed,
        "low_drift_highpass_correlation_eeg": low_drift_correlation,
        "low_drift_highpass_rrmse_eeg": low_drift_rrmse,
        "event_checked_eeg": event_checked,
        "event_failed_eeg": event_failed,
        "event_outside_correlation_eeg": event_outside_correlation,
        "event_outside_rrmse_eeg": event_outside_rrmse,
        "ecg_checked_eeg": ecg_checked,
        "ecg_failed_eeg": ecg_failed,
        "ecg_input_abs_correlation_eeg": ecg_input_abs_correlation,
        "ecg_output_abs_correlation_eeg": ecg_output_abs_correlation,
        "parameters": {
            "preservation_correlation_min": policy.preservation_correlation_min,
            "preservation_rrmse_max": policy.preservation_rrmse_max,
            "low_drift_highpass_hz": policy.low_drift_highpass_hz,
            "edge_trim_sec": policy.edge_trim_sec,
            "event_z_threshold": policy.event_z_threshold,
            "event_padding_sec": policy.event_padding_sec,
            "ecg_nonworsening_tolerance": policy.ecg_nonworsening_tolerance,
        },
    }


def apply_output_safety_fallback(
    *,
    input_eeg: np.ndarray,
    candidate_eeg: np.ndarray,
    process_eeg: np.ndarray,
    sample_rate: float,
) -> dict[str, Any]:
    """Reject unsafe channel-wise model outputs under original thresholds.

    The thresholds are derived once from the unchanged recording. Only channel
    windows that actually passed through the model are checked. If the input
    is already FLATLINE/CLIPPING according to these reconstructed thresholds,
    or the candidate newly introduces either marker, the unchanged input is
    returned for that channel window.
    """

    input_eeg = np.asarray(input_eeg, dtype=np.float32)
    candidate_eeg = np.asarray(candidate_eeg, dtype=np.float32)
    process_eeg = np.asarray(process_eeg, dtype=bool)
    if input_eeg.ndim != 3 or input_eeg.shape[1] != 2:
        raise ValueError(
            "input_eeg must have shape [windows, 2, samples], "
            f"got {input_eeg.shape}."
        )
    if candidate_eeg.shape != input_eeg.shape:
        raise ValueError(
            "candidate_eeg must match input_eeg, got "
            f"{candidate_eeg.shape} and {input_eeg.shape}."
        )
    if process_eeg.shape != input_eeg.shape[:2]:
        raise ValueError(
            f"process_eeg must have shape {input_eeg.shape[:2]}, "
            f"got {process_eeg.shape}."
        )
    sample_rate = float(sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be a positive finite value.")

    thresholds = derive_recording_thresholds(input_eeg, sample_rate)
    checked = process_eeg.copy()
    input_unsafe = np.zeros_like(checked)
    new_flatline = np.zeros_like(checked)
    new_clipping = np.zeros_like(checked)
    fallback = np.zeros_like(checked)
    safe_eeg = candidate_eeg.copy()

    flattened_indices = np.flatnonzero(checked.reshape(-1))
    if len(flattened_indices):
        samples = input_eeg.shape[-1]
        flattened_input = input_eeg.reshape(-1, samples)
        flattened_candidate = candidate_eeg.reshape(-1, samples)
        input_features = signal_features(
            flattened_input[flattened_indices],
            sample_rate,
        )
        output_features = signal_features(
            flattened_candidate[flattened_indices],
            sample_rate,
        )

        input_flatline = (
            input_features["std"] <= thresholds["threshold_flatline_std"]
        ) | (
            input_features["ptp"] <= thresholds["threshold_flatline_ptp"]
        )
        output_flatline = (
            output_features["std"] <= thresholds["threshold_flatline_std"]
        ) | (
            output_features["ptp"] <= thresholds["threshold_flatline_ptp"]
        )
        input_clipping = (
            input_features["clipping_ratio"]
            > thresholds["threshold_clipping_ratio"]
        )
        output_clipping = (
            output_features["clipping_ratio"]
            > thresholds["threshold_clipping_ratio"]
        )

        local_input_unsafe = input_flatline | input_clipping
        local_new_flatline = output_flatline & ~input_flatline
        local_new_clipping = output_clipping & ~input_clipping
        local_fallback = (
            local_input_unsafe | local_new_flatline | local_new_clipping
        )

        input_unsafe.reshape(-1)[flattened_indices] = local_input_unsafe
        new_flatline.reshape(-1)[flattened_indices] = local_new_flatline
        new_clipping.reshape(-1)[flattened_indices] = local_new_clipping
        fallback.reshape(-1)[flattened_indices] = local_fallback
        fallback_indices = flattened_indices[local_fallback]
        safe_eeg.reshape(-1, samples)[fallback_indices] = flattened_input[
            fallback_indices
        ]

    return {
        "denoised_eeg": safe_eeg,
        "checked_eeg": checked,
        "input_unsafe_eeg": input_unsafe,
        "new_flatline_eeg": new_flatline,
        "new_clipping_eeg": new_clipping,
        "fallback_eeg": fallback,
        "thresholds": thresholds,
    }


def _condition_matrix(
    record: dict[str, np.ndarray],
    marker_names: tuple[str, ...],
) -> np.ndarray:
    exported_names = [str(value) for value in record["marker_names"].tolist()]
    exported = np.asarray(record["markers"], dtype=np.float32)
    condition = np.zeros(
        (exported.shape[0], exported.shape[1], len(marker_names)),
        dtype=np.float32,
    )
    for target_index, name in enumerate(marker_names):
        if name in exported_names:
            condition[:, :, target_index] = exported[
                :, :, exported_names.index(name)
            ]

    if "CLEAN" in marker_names:
        if "strict_clean_eeg" not in record:
            raise KeyError(
                "Conditioned inference requires channel-specific "
                "strict_clean_eeg labels to set the CLEAN condition."
            )
        strict_clean = np.asarray(record["strict_clean_eeg"], dtype=np.float32)
        expected_shape = exported.shape[:2]
        if strict_clean.shape != expected_shape:
            raise ValueError(
                "strict_clean_eeg must have shape "
                f"{expected_shape}, got {strict_clean.shape}."
            )
        condition[:, :, marker_names.index("CLEAN")] = strict_clean
    return condition


def build_inference_plan(
    *,
    record: dict[str, np.ndarray],
    marker_names: tuple[str, ...],
    policy: InferencePolicy,
) -> dict[str, np.ndarray]:
    """Build channel-wise masks and conditions without requiring PyTorch."""

    eeg = np.asarray(record["eeg"], dtype=np.float32)
    if eeg.ndim != 3 or eeg.shape[1] != 2:
        raise ValueError(f"Expected EEG shape [windows, 2, samples], got {eeg.shape}.")
    windows, channels, samples = eeg.shape
    conditions = _condition_matrix(record, marker_names)
    exported_names = [str(value) for value in record["marker_names"].tolist()]
    exported_markers = np.asarray(record["markers"], dtype=bool)
    if exported_markers.shape != (windows, channels, len(exported_names)):
        raise ValueError(
            f"Marker tensor has incompatible shape {exported_markers.shape}."
        )

    strict = np.asarray(record["strict_clean_eeg"], dtype=bool)
    if strict.shape != (windows, channels):
        raise ValueError(
            f"strict_clean_eeg must have shape {(windows, channels)}, "
            f"got {strict.shape}."
        )
    preserved_strict = strict if policy.preserve_strict_clean else np.zeros_like(strict)

    bypassed = np.zeros((windows, channels), dtype=bool)
    for marker_name in policy.bypass_markers:
        if marker_name in exported_names:
            bypassed |= exported_markers[
                :, :, exported_names.index(marker_name)
            ]
    process = ~(preserved_strict | bypassed)
    low_information = detect_low_information_inputs(
        eeg,
        policy.low_information_relative_floor,
    )
    low_information_candidate = low_information["candidate_eeg"]
    low_information_bypassed = (
        process & low_information_candidate
        if policy.selective_preservation_fallback
        else np.zeros_like(process)
    )
    process &= ~low_information_bypassed

    ecg_marked = np.zeros((windows, channels), dtype=bool)
    if "ECG" in exported_names:
        ecg_marked = exported_markers[:, :, exported_names.index("ECG")]

    ecg_correlation = np.full((windows, channels), np.nan, dtype=np.float32)
    ecg_available = bool(np.asarray(record["ecg_available"]).item())
    if ecg_available:
        ecg = np.asarray(record["ecg"], dtype=np.float32)
        if ecg.shape != (windows, 1, samples):
            raise ValueError(
                f"ECG must have shape {(windows, 1, samples)}, got {ecg.shape}."
            )
        repeated_ecg = np.repeat(ecg, channels, axis=1)
        correlations = _row_correlation(
            eeg.reshape(-1, samples),
            repeated_ecg.reshape(-1, samples),
        ).reshape(windows, channels)
        ecg_correlation[:] = correlations.astype(np.float32)

    ecg_active = (
        ecg_marked
        & process
        & np.isfinite(ecg_correlation)
        & (np.abs(ecg_correlation) >= policy.ecg_min_abs_correlation)
    )
    ecg_suppressed = ecg_marked & ~ecg_active

    # The model must not see a binary ECG instruction when the corresponding
    # reference was suppressed.  Other co-occurring marker conditions remain.
    if "ECG" in marker_names:
        conditions[:, :, marker_names.index("ECG")] = ecg_active.astype(np.float32)

    return {
        "conditions": conditions,
        "process_eeg": process,
        "preserved_strict_clean_eeg": preserved_strict,
        "bypassed_marker_eeg": bypassed,
        "low_information_candidate_eeg": low_information_candidate,
        "low_information_bypassed_eeg": low_information_bypassed,
        "low_information_centered_rms_eeg": low_information[
            "centered_rms_eeg"
        ],
        "low_information_peak_to_peak_eeg": low_information[
            "peak_to_peak_eeg"
        ],
        "low_information_rms_floor_by_channel": low_information[
            "rms_floor_by_channel"
        ],
        "low_information_peak_to_peak_floor_by_channel": low_information[
            "peak_to_peak_floor_by_channel"
        ],
        "ecg_marked_eeg": ecg_marked,
        "ecg_reference_active_eeg": ecg_active,
        "ecg_reference_suppressed_eeg": ecg_suppressed,
        "ecg_input_correlation_eeg": ecg_correlation,
    }


def apply_loaded_model_to_recording(
    *,
    model: Any,
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    policy: InferencePolicy | None = None,
    batch_size: int = 256,
    device: Any,
) -> Path:
    """Apply an already loaded model under one explicit inference policy."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if policy is None:
        policy = InferencePolicy()

    # PyTorch stays optional for plan inspection and evaluation-only workflows.
    import torch

    record = load_dose_recording_npz(input_path)
    eeg = np.asarray(record["eeg"], dtype=np.float32)
    ecg = np.asarray(record["ecg"], dtype=np.float32)[:, 0]
    marker_names = tuple(checkpoint["model_config"]["marker_names"])
    plan = build_inference_plan(
        record=record,
        marker_names=marker_names,
        policy=policy,
    )

    windows, channels, samples = eeg.shape
    flattened = eeg.reshape(-1, samples)
    denoised = flattened.copy()
    flattened_conditions = plan["conditions"].reshape(-1, len(marker_names))
    process_indices = np.flatnonzero(plan["process_eeg"].reshape(-1))
    repeated_ecg = np.repeat(ecg[:, None, :], channels, axis=1).reshape(
        -1, samples
    )
    flattened_ecg_active = plan["ecg_reference_active_eeg"].reshape(-1)

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(process_indices), batch_size):
            indices = process_indices[start : start + batch_size]
            values = flattened[indices]
            location = values.mean(axis=1, keepdims=True)
            scale = values.std(axis=1, keepdims=True)
            scale[~np.isfinite(scale) | (scale < EPSILON)] = 1.0
            normalised = (values - location) / scale

            reference = repeated_ecg[indices].copy()
            reference = reference - reference.mean(axis=1, keepdims=True)
            reference_scale = reference.std(axis=1, keepdims=True)
            reference_scale[
                ~np.isfinite(reference_scale) | (reference_scale < EPSILON)
            ] = 1.0
            reference = reference / reference_scale
            reference *= flattened_ecg_active[indices, None]

            prediction = model(
                torch.from_numpy(normalised[:, None, :]).to(device),
                reference=torch.from_numpy(reference[:, None, :]).to(device),
                markers=torch.from_numpy(flattened_conditions[indices]).to(device),
            )
            restored = prediction[:, 0].float().cpu().numpy() * scale + location
            denoised[indices] = restored.astype(np.float32)

    denoised_eeg = denoised.reshape(eeg.shape)
    sample_rate = float(np.asarray(record["sample_rate"]).item())
    safety_shape = eeg.shape[:2]
    safety = {
        "checked_eeg": np.zeros(safety_shape, dtype=bool),
        "input_unsafe_eeg": np.zeros(safety_shape, dtype=bool),
        "new_flatline_eeg": np.zeros(safety_shape, dtype=bool),
        "new_clipping_eeg": np.zeros(safety_shape, dtype=bool),
        "fallback_eeg": np.zeros(safety_shape, dtype=bool),
        "thresholds": {},
    }
    if policy.output_safety_fallback:
        safety = apply_output_safety_fallback(
            input_eeg=eeg,
            candidate_eeg=denoised_eeg,
            process_eeg=plan["process_eeg"],
            sample_rate=sample_rate,
        )
        denoised_eeg = safety["denoised_eeg"]
    selective = {
        "denoised_eeg": denoised_eeg,
        "checked_eeg": np.zeros(safety_shape, dtype=bool),
        "fallback_eeg": np.zeros(safety_shape, dtype=bool),
        "low_drift_checked_eeg": np.zeros(safety_shape, dtype=bool),
        "low_drift_failed_eeg": np.zeros(safety_shape, dtype=bool),
        "low_drift_highpass_correlation_eeg": np.full(
            safety_shape, np.nan, dtype=np.float32
        ),
        "low_drift_highpass_rrmse_eeg": np.full(
            safety_shape, np.nan, dtype=np.float32
        ),
        "event_checked_eeg": np.zeros(safety_shape, dtype=bool),
        "event_failed_eeg": np.zeros(safety_shape, dtype=bool),
        "event_outside_correlation_eeg": np.full(
            safety_shape, np.nan, dtype=np.float32
        ),
        "event_outside_rrmse_eeg": np.full(
            safety_shape, np.nan, dtype=np.float32
        ),
        "ecg_checked_eeg": np.zeros(safety_shape, dtype=bool),
        "ecg_failed_eeg": np.zeros(safety_shape, dtype=bool),
        "ecg_input_abs_correlation_eeg": np.full(
            safety_shape, np.nan, dtype=np.float32
        ),
        "ecg_output_abs_correlation_eeg": np.full(
            safety_shape, np.nan, dtype=np.float32
        ),
        "parameters": {},
    }
    if policy.selective_preservation_fallback:
        selective = apply_selective_preservation_fallback(
            record=record,
            input_eeg=eeg,
            candidate_eeg=denoised_eeg,
            process_eeg=plan["process_eeg"],
            output_safety_fallback_eeg=safety["fallback_eeg"],
            ecg_reference_active_eeg=plan["ecg_reference_active_eeg"],
            sample_rate=sample_rate,
            policy=policy,
        )
        denoised_eeg = selective["denoised_eeg"]
    difference = denoised_eeg - eeg
    change_rmse = np.sqrt(np.mean(np.square(difference), axis=2)).astype(np.float32)

    output_payload = dict(record)
    output_payload.update(
        {
            "denoised_eeg": denoised_eeg,
            "input_to_output_rmse": change_rmse,
            "autoencoder_checkpoint": np.array(
                str(Path(checkpoint_path).expanduser().resolve())
            ),
            "autoencoder_variant": np.array(checkpoint["model_config"]["variant"]),
            "autoencoder_stage": np.array(str(checkpoint.get("stage", "unknown"))),
            "inference_policy_version": np.array(policy.version),
            "inference_policy_json": np.array(policy_json(policy)),
            "inference_processed_eeg": plan["process_eeg"].astype(np.int8),
            "inference_preserved_strict_clean_eeg": plan[
                "preserved_strict_clean_eeg"
            ].astype(np.int8),
            "inference_bypassed_marker_eeg": plan[
                "bypassed_marker_eeg"
            ].astype(np.int8),
            "inference_low_information_candidate_eeg": plan[
                "low_information_candidate_eeg"
            ].astype(np.int8),
            "inference_low_information_bypassed_eeg": plan[
                "low_information_bypassed_eeg"
            ].astype(np.int8),
            "inference_low_information_centered_rms_eeg": plan[
                "low_information_centered_rms_eeg"
            ].astype(np.float32),
            "inference_low_information_peak_to_peak_eeg": plan[
                "low_information_peak_to_peak_eeg"
            ].astype(np.float32),
            "inference_low_information_rms_floor_by_channel": plan[
                "low_information_rms_floor_by_channel"
            ].astype(np.float32),
            "inference_low_information_peak_to_peak_floor_by_channel": plan[
                "low_information_peak_to_peak_floor_by_channel"
            ].astype(np.float32),
            "inference_low_information_bypassed_count": np.array(
                np.count_nonzero(plan["low_information_bypassed_eeg"]),
                dtype=np.int64,
            ),
            "inference_ecg_marked_eeg": plan["ecg_marked_eeg"].astype(np.int8),
            "inference_ecg_reference_active_eeg": plan[
                "ecg_reference_active_eeg"
            ].astype(np.int8),
            "inference_ecg_reference_suppressed_eeg": plan[
                "ecg_reference_suppressed_eeg"
            ].astype(np.int8),
            "inference_ecg_input_correlation_eeg": plan[
                "ecg_input_correlation_eeg"
            ].astype(np.float32),
            "inference_output_safety_checked_eeg": safety[
                "checked_eeg"
            ].astype(np.int8),
            "inference_output_safety_input_unsafe_eeg": safety[
                "input_unsafe_eeg"
            ].astype(np.int8),
            "inference_output_safety_new_flatline_eeg": safety[
                "new_flatline_eeg"
            ].astype(np.int8),
            "inference_output_safety_new_clipping_eeg": safety[
                "new_clipping_eeg"
            ].astype(np.int8),
            "inference_output_safety_fallback_eeg": safety[
                "fallback_eeg"
            ].astype(np.int8),
            "inference_output_safety_checked_count": np.array(
                np.count_nonzero(safety["checked_eeg"]),
                dtype=np.int64,
            ),
            "inference_output_safety_fallback_count": np.array(
                np.count_nonzero(safety["fallback_eeg"]),
                dtype=np.int64,
            ),
            "inference_output_safety_new_flatline_count": np.array(
                np.count_nonzero(safety["new_flatline_eeg"]),
                dtype=np.int64,
            ),
            "inference_output_safety_new_clipping_count": np.array(
                np.count_nonzero(safety["new_clipping_eeg"]),
                dtype=np.int64,
            ),
            "inference_output_safety_thresholds_json": np.array(
                json.dumps(
                    safety["thresholds"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "inference_selective_safety_checked_eeg": selective[
                "checked_eeg"
            ].astype(np.int8),
            "inference_selective_safety_fallback_eeg": selective[
                "fallback_eeg"
            ].astype(np.int8),
            "inference_selective_low_drift_checked_eeg": selective[
                "low_drift_checked_eeg"
            ].astype(np.int8),
            "inference_selective_low_drift_failed_eeg": selective[
                "low_drift_failed_eeg"
            ].astype(np.int8),
            "inference_selective_low_drift_highpass_correlation_eeg": selective[
                "low_drift_highpass_correlation_eeg"
            ].astype(np.float32),
            "inference_selective_low_drift_highpass_rrmse_eeg": selective[
                "low_drift_highpass_rrmse_eeg"
            ].astype(np.float32),
            "inference_selective_event_checked_eeg": selective[
                "event_checked_eeg"
            ].astype(np.int8),
            "inference_selective_event_failed_eeg": selective[
                "event_failed_eeg"
            ].astype(np.int8),
            "inference_selective_event_outside_correlation_eeg": selective[
                "event_outside_correlation_eeg"
            ].astype(np.float32),
            "inference_selective_event_outside_rrmse_eeg": selective[
                "event_outside_rrmse_eeg"
            ].astype(np.float32),
            "inference_selective_ecg_checked_eeg": selective[
                "ecg_checked_eeg"
            ].astype(np.int8),
            "inference_selective_ecg_failed_eeg": selective[
                "ecg_failed_eeg"
            ].astype(np.int8),
            "inference_selective_ecg_input_abs_correlation_eeg": selective[
                "ecg_input_abs_correlation_eeg"
            ].astype(np.float32),
            "inference_selective_ecg_output_abs_correlation_eeg": selective[
                "ecg_output_abs_correlation_eeg"
            ].astype(np.float32),
            "inference_selective_safety_checked_count": np.array(
                np.count_nonzero(selective["checked_eeg"]),
                dtype=np.int64,
            ),
            "inference_selective_safety_fallback_count": np.array(
                np.count_nonzero(selective["fallback_eeg"]),
                dtype=np.int64,
            ),
            "inference_selective_low_drift_failed_count": np.array(
                np.count_nonzero(selective["low_drift_failed_eeg"]),
                dtype=np.int64,
            ),
            "inference_selective_event_failed_count": np.array(
                np.count_nonzero(selective["event_failed_eeg"]),
                dtype=np.int64,
            ),
            "inference_selective_ecg_failed_count": np.array(
                np.count_nonzero(selective["ecg_failed_eeg"]),
                dtype=np.int64,
            ),
            "inference_selective_safety_parameters_json": np.array(
                json.dumps(
                    selective["parameters"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **output_payload)
    temporary_path.replace(output_path)
    return output_path


def apply_checkpoint_to_recording(
    *,
    input_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    policy: InferencePolicy | None = None,
    batch_size: int = 256,
    device_name: str = "auto",
) -> Path:
    """Load one checkpoint and apply it under an explicit policy."""

    from .checkpoints import load_model
    from .runtime import resolve_device

    device = resolve_device(device_name)
    model, checkpoint = load_model(checkpoint_path, device=device)
    return apply_loaded_model_to_recording(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        output_path=output_path,
        policy=policy,
        batch_size=batch_size,
        device=device,
    )
