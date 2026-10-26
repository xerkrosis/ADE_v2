"""Validated bridge from autoencoder prediction NPZ files into ADE pipeline.

The main preprocessing pipeline works on continuous signals, whereas the
autoencoder deliberately works on non-overlapping windows.  This module checks
that a prediction file belongs to the current recording and reconstructs a
continuous signal by replacing only channel windows whose model output is
accepted by the inference policy. Policy-preserved windows, low-information,
output-safety and selective-preservation fallbacks, plus any short tail that
was not windowed, retain the current pipeline's original Float64 samples.
When configured, the additional v2.5 hybrid provenance and gates are checked
before its signal is admitted to ADE.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AutoencoderPredictionMetadata:
    """Validated provenance returned together with the reconstructed signal."""

    prediction_file: str
    checkpoint: str
    model_variant: str
    model_stage: str
    policy_version: str
    ecg_min_abs_correlation: float
    processed_channel_windows: int
    preserved_strict_clean_channel_windows: int
    bypassed_marker_channel_windows: int
    output_safety_fallback_channel_windows: int
    output_safety_new_flatline_channel_windows: int
    output_safety_new_clipping_channel_windows: int
    low_information_bypassed_channel_windows: int
    selective_preservation_fallback_channel_windows: int
    hybrid_policy_version: str
    hybrid_selected_v21_channel_windows: int


def _scalar(npz: np.lib.npyio.NpzFile, key: str):
    if key not in npz.files:
        raise KeyError(f"Autoencoder prediction is missing {key!r}.")
    value = np.asarray(npz[key])
    if value.ndim != 0:
        raise ValueError(f"Autoencoder field {key!r} must be scalar.")
    return value.item()


def _require_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(
            f"Autoencoder field {name!r} has shape {value.shape}; "
            f"expected {expected}."
        )


def _validate_policy_subset(
    stored_policy: Mapping[str, object],
    expected_policy: Mapping[str, object],
) -> None:
    """Require every configured policy value without overfitting to metadata."""
    missing = sorted(set(expected_policy).difference(stored_policy))
    if missing:
        raise KeyError(
            "Autoencoder hybrid policy is missing: " + ", ".join(missing)
        )

    for key, expected in expected_policy.items():
        actual = stored_policy[key]
        if isinstance(expected, bool):
            matches = isinstance(actual, bool) and actual is expected
        elif isinstance(expected, (int, float)):
            try:
                matches = np.isclose(
                    float(actual),
                    float(expected),
                    rtol=0.0,
                    atol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
        elif isinstance(expected, (tuple, list)):
            matches = list(actual) == list(expected) if isinstance(
                actual, (tuple, list)
            ) else False
        else:
            matches = actual == expected

        if not matches:
            raise ValueError(
                f"Autoencoder hybrid policy {key!r} is {actual!r}; "
                f"expected {expected!r}."
            )


def _window_view(
    signal1: np.ndarray,
    signal2: np.ndarray,
    *,
    window_size: int,
) -> np.ndarray:
    """Reproduce the current pipeline's zero-overlap window convention."""
    starts = range(0, len(signal1) - window_size, window_size)
    windows = [
        np.stack(
            [signal1[start : start + window_size], signal2[start : start + window_size]],
            axis=0,
        )
        for start in starts
    ]
    if not windows:
        return np.empty((0, 2, window_size), dtype=np.float64)
    return np.stack(windows, axis=0)


def load_autoencoder_prediction_signal(
    *,
    prediction_path: str | Path,
    recording_id: str,
    original_eeg1: np.ndarray,
    original_eeg2: np.ndarray,
    sample_rate: float,
    window_seconds: float,
    window_overlap: float,
    expected_policy_version: str = "v2.1",
    expected_ecg_min_abs_correlation: float = 0.05,
    expected_bypass_markers: Iterable[str] = ("FLATLINE", "CLIPPING"),
    require_strict_clean_preservation: bool = True,
    expected_hybrid_policy: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, AutoencoderPredictionMetadata]:
    """Validate one prediction NPZ and reconstruct two continuous EEG channels.

    The integration intentionally supports only the experiment's established
    five-second, zero-overlap setting.  Supporting overlap would require an
    explicit overlap-add policy and must not be guessed silently.
    """
    path = Path(prediction_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Autoencoder prediction not found: {path}")

    eeg1 = np.asarray(original_eeg1, dtype=np.float64)
    eeg2 = np.asarray(original_eeg2, dtype=np.float64)
    if eeg1.ndim != 1 or eeg2.ndim != 1 or eeg1.shape != eeg2.shape:
        raise ValueError("Original EEG1 and EEG2 must be equally sized 1-D arrays.")
    if not np.isclose(float(window_overlap), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            "Autoencoder pipeline integration currently requires window_overlap=0.0."
        )

    window_size = int(float(window_seconds) * float(sample_rate))
    if window_size <= 0:
        raise ValueError("window_seconds * sample_rate must be positive.")
    expected_source = _window_view(eeg1, eeg2, window_size=window_size)

    with np.load(path, allow_pickle=False) as npz:
        required = {
            "recording_id",
            "sample_rate",
            "window_seconds",
            "window_overlap",
            "window_number",
            "eeg",
            "denoised_eeg",
            "autoencoder_checkpoint",
            "autoencoder_variant",
            "autoencoder_stage",
            "inference_policy_version",
            "inference_policy_json",
            "inference_processed_eeg",
            "inference_preserved_strict_clean_eeg",
            "inference_bypassed_marker_eeg",
        }
        missing = sorted(required.difference(npz.files))
        if missing:
            raise KeyError(
                "Autoencoder prediction is missing required fields: "
                + ", ".join(missing)
            )

        stored_recording = str(_scalar(npz, "recording_id"))
        if stored_recording != str(recording_id):
            raise ValueError(
                f"Prediction recording_id={stored_recording!r}, "
                f"but the current recording is {recording_id!r}."
            )

        numeric_checks = {
            "sample_rate": float(sample_rate),
            "window_seconds": float(window_seconds),
            "window_overlap": float(window_overlap),
        }
        for key, expected in numeric_checks.items():
            actual = float(_scalar(npz, key))
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"Prediction {key}={actual:g}, expected {expected:g}."
                )

        source = np.asarray(npz["eeg"], dtype=np.float64)
        denoised = np.asarray(npz["denoised_eeg"], dtype=np.float64)
        _require_shape("eeg", source, expected_source.shape)
        _require_shape("denoised_eeg", denoised, expected_source.shape)
        if not np.isfinite(source).all() or not np.isfinite(denoised).all():
            raise ValueError("Autoencoder source or output contains non-finite values.")

        if not np.allclose(source, expected_source, rtol=1e-6, atol=1e-5):
            max_difference = float(np.max(np.abs(source - expected_source)))
            raise ValueError(
                "Prediction source EEG does not match the current preprocessed "
                f"recording (maximum absolute difference {max_difference:.6g})."
            )

        window_numbers = np.asarray(npz["window_number"], dtype=np.int64)
        expected_numbers = np.arange(len(expected_source), dtype=np.int64)
        if not np.array_equal(window_numbers, expected_numbers):
            raise ValueError("Prediction window_number is not the expected 0..N-1 sequence.")

        policy_version = str(_scalar(npz, "inference_policy_version"))
        if policy_version != str(expected_policy_version):
            raise ValueError(
                f"Prediction policy is {policy_version!r}; "
                f"expected {expected_policy_version!r}."
            )
        try:
            policy = json.loads(str(_scalar(npz, "inference_policy_json")))
        except json.JSONDecodeError as error:
            raise ValueError("inference_policy_json is not valid JSON.") from error

        threshold = float(policy.get("ecg_min_abs_correlation", np.nan))
        if not np.isclose(
            threshold,
            float(expected_ecg_min_abs_correlation),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Prediction ECG activation threshold is "
                f"{threshold:g}; expected {expected_ecg_min_abs_correlation:g}."
            )
        if bool(policy.get("preserve_strict_clean")) != bool(
            require_strict_clean_preservation
        ):
            raise ValueError(
                "Prediction strict-clean preservation policy does not match "
                "the requested integration policy."
            )
        stored_bypass = set(policy.get("bypass_markers", []))
        required_bypass = set(expected_bypass_markers)
        if stored_bypass != required_bypass:
            raise ValueError(
                f"Prediction bypass markers are {sorted(stored_bypass)}, "
                f"expected {sorted(required_bypass)}."
            )

        output_safety_enabled = bool(
            policy.get("output_safety_fallback", False)
        )
        selective_preservation_enabled = bool(
            policy.get("selective_preservation_fallback", False)
        )
        if str(expected_policy_version) == "v2.2" and not output_safety_enabled:
            raise ValueError(
                "Inference policy v2.2 requires output_safety_fallback=True."
            )
        if str(expected_policy_version) == "v2.3" and (
            not output_safety_enabled
            or not selective_preservation_enabled
        ):
            raise ValueError(
                "Inference policy v2.3 requires output-safety and selective "
                "preservation fallbacks."
            )

        preserved = np.asarray(
            npz["inference_preserved_strict_clean_eeg"], dtype=bool
        )
        bypassed = np.asarray(npz["inference_bypassed_marker_eeg"], dtype=bool)
        processed = np.asarray(npz["inference_processed_eeg"], dtype=bool)
        decision_shape = expected_source.shape[:2]
        _require_shape("inference_processed_eeg", processed, decision_shape)
        _require_shape(
            "inference_preserved_strict_clean_eeg", preserved, decision_shape
        )
        _require_shape("inference_bypassed_marker_eeg", bypassed, decision_shape)

        low_information_bypassed = np.zeros(decision_shape, dtype=bool)
        if "inference_low_information_bypassed_eeg" in npz.files:
            low_information_bypassed = np.asarray(
                npz["inference_low_information_bypassed_eeg"],
                dtype=bool,
            )
            _require_shape(
                "inference_low_information_bypassed_eeg",
                low_information_bypassed,
                decision_shape,
            )
        elif selective_preservation_enabled:
            raise KeyError(
                "Selective prediction is missing "
                "'inference_low_information_bypassed_eeg'."
            )

        selective_fallback = np.zeros(decision_shape, dtype=bool)
        if "inference_selective_safety_fallback_eeg" in npz.files:
            selective_fallback = np.asarray(
                npz["inference_selective_safety_fallback_eeg"],
                dtype=bool,
            )
            _require_shape(
                "inference_selective_safety_fallback_eeg",
                selective_fallback,
                decision_shape,
            )
        elif selective_preservation_enabled:
            raise KeyError(
                "Selective prediction is missing "
                "'inference_selective_safety_fallback_eeg'."
            )

        primary_copy_mask = (
            preserved
            | bypassed
            | low_information_bypassed
        )
        if np.any(processed & primary_copy_mask):
            raise ValueError(
                "Prediction marks primary copied and processed states "
                "simultaneously."
            )
        if np.any(~(processed | primary_copy_mask)):
            raise ValueError(
                "Prediction contains channel windows without a primary "
                "policy action."
            )

        safety_fallback = np.zeros(decision_shape, dtype=bool)
        safety_new_flatline = np.zeros(decision_shape, dtype=bool)
        safety_new_clipping = np.zeros(decision_shape, dtype=bool)
        if output_safety_enabled:
            safety_fields = {
                "inference_output_safety_checked_eeg",
                "inference_output_safety_fallback_eeg",
                "inference_output_safety_new_flatline_eeg",
                "inference_output_safety_new_clipping_eeg",
            }
            missing_safety = sorted(safety_fields.difference(npz.files))
            if missing_safety:
                raise KeyError(
                    "Safety-gated prediction is missing required fields: "
                    + ", ".join(missing_safety)
                )
            safety_checked = np.asarray(
                npz["inference_output_safety_checked_eeg"],
                dtype=bool,
            )
            safety_fallback = np.asarray(
                npz["inference_output_safety_fallback_eeg"],
                dtype=bool,
            )
            safety_new_flatline = np.asarray(
                npz["inference_output_safety_new_flatline_eeg"],
                dtype=bool,
            )
            safety_new_clipping = np.asarray(
                npz["inference_output_safety_new_clipping_eeg"],
                dtype=bool,
            )
            for field_name, field_value in (
                ("inference_output_safety_checked_eeg", safety_checked),
                ("inference_output_safety_fallback_eeg", safety_fallback),
                (
                    "inference_output_safety_new_flatline_eeg",
                    safety_new_flatline,
                ),
                (
                    "inference_output_safety_new_clipping_eeg",
                    safety_new_clipping,
                ),
            ):
                _require_shape(field_name, field_value, decision_shape)
            if not np.array_equal(safety_checked, processed):
                raise ValueError(
                    "Output-safety checked mask must equal the model process mask."
                )
            if np.any(safety_fallback & ~processed):
                raise ValueError(
                    "Output-safety fallback contains an unprocessed channel window."
                )
            if np.any(
                (safety_new_flatline | safety_new_clipping)
                & ~safety_fallback
            ):
                raise ValueError(
                    "A newly unsafe output is not covered by the fallback mask."
                )
            if not np.array_equal(
                denoised[safety_fallback],
                source[safety_fallback],
            ):
                raise ValueError(
                    "Output-safety fallback windows are not exact input copies."
                )

        if np.any(selective_fallback & ~processed):
            raise ValueError(
                "Selective preservation fallback contains an unprocessed "
                "channel window."
            )

        exact_copy_mask = (
            primary_copy_mask
            | safety_fallback
            | selective_fallback
        )
        if not np.array_equal(
            denoised[exact_copy_mask],
            source[exact_copy_mask],
        ):
            raise ValueError(
                "Prediction violates an exact-copy preservation or safety "
                "fallback."
            )

        accepted_model_output = (
            processed
            & ~safety_fallback
            & ~selective_fallback
        )

        hybrid_policy_version = ""
        hybrid_selected = np.zeros(decision_shape, dtype=bool)
        contains_hybrid_policy = "hybrid_policy_version" in npz.files
        if contains_hybrid_policy:
            hybrid_policy_version = str(
                _scalar(npz, "hybrid_policy_version")
            )
        if expected_hybrid_policy is not None:
            if not isinstance(expected_hybrid_policy, Mapping):
                raise TypeError(
                    "expected_hybrid_policy must be a mapping or None."
                )
            if not expected_hybrid_policy:
                raise ValueError("expected_hybrid_policy must not be empty.")
            hybrid_fields = {
                "hybrid_policy_version",
                "hybrid_policy_json",
                "hybrid_v21_selected_eeg",
                "hybrid_v21_selected_count",
                "hybrid_v21_v24_reference_selected_eeg",
                "hybrid_v21_absolute_power_ratio_4_30_eeg",
                "hybrid_v21_absolute_power_4_30_pass_eeg",
                "hybrid_v21_slow_corr_gate_pass_eeg",
                "hybrid_v3_slow_corr_window",
                "hybrid_final_slow_corr_window",
            }
            missing_hybrid = sorted(hybrid_fields.difference(npz.files))
            if missing_hybrid:
                raise KeyError(
                    "Autoencoder v2.5 hybrid prediction is missing: "
                    + ", ".join(missing_hybrid)
                )

            expected_hybrid_version = str(
                expected_hybrid_policy.get("version", "")
            )
            if not expected_hybrid_version:
                raise ValueError(
                    "expected_hybrid_policy must contain a non-empty "
                    "'version'."
                )
            if hybrid_policy_version != expected_hybrid_version:
                raise ValueError(
                    f"Hybrid policy is {hybrid_policy_version!r}; expected "
                    f"{expected_hybrid_version!r}."
                )
            try:
                hybrid_policy = json.loads(
                    str(_scalar(npz, "hybrid_policy_json"))
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    "hybrid_policy_json is not valid JSON."
                ) from error
            if not isinstance(hybrid_policy, dict):
                raise TypeError("hybrid_policy_json must contain an object.")
            _validate_policy_subset(
                hybrid_policy,
                expected_hybrid_policy,
            )

            hybrid_selected = np.asarray(
                npz["hybrid_v21_selected_eeg"], dtype=bool
            )
            hybrid_reference = np.asarray(
                npz["hybrid_v21_v24_reference_selected_eeg"],
                dtype=bool,
            )
            hybrid_power_pass = np.asarray(
                npz["hybrid_v21_absolute_power_4_30_pass_eeg"],
                dtype=bool,
            )
            hybrid_power_ratio = np.asarray(
                npz["hybrid_v21_absolute_power_ratio_4_30_eeg"],
                dtype=float,
            )
            hybrid_slow_corr_pass = np.asarray(
                npz["hybrid_v21_slow_corr_gate_pass_eeg"],
                dtype=bool,
            )
            for field_name, field_value in (
                ("hybrid_v21_selected_eeg", hybrid_selected),
                (
                    "hybrid_v21_v24_reference_selected_eeg",
                    hybrid_reference,
                ),
                (
                    "hybrid_v21_absolute_power_4_30_pass_eeg",
                    hybrid_power_pass,
                ),
                (
                    "hybrid_v21_absolute_power_ratio_4_30_eeg",
                    hybrid_power_ratio,
                ),
                (
                    "hybrid_v21_slow_corr_gate_pass_eeg",
                    hybrid_slow_corr_pass,
                ),
            ):
                _require_shape(field_name, field_value, decision_shape)
            if np.any(
                hybrid_selected
                & ~(
                    hybrid_reference
                    & hybrid_power_pass
                    & hybrid_slow_corr_pass
                )
            ):
                raise ValueError(
                    "Hybrid prediction selects a v2.1 output outside the "
                    "configured v2.5 gates."
                )
            stored_selected_count = int(
                _scalar(npz, "hybrid_v21_selected_count")
            )
            if stored_selected_count != int(hybrid_selected.sum()):
                raise ValueError(
                    "Hybrid selected-count metadata does not match its mask."
                )
            if "absolute_power_ratio_min" in expected_hybrid_policy:
                minimum_power_ratio = float(
                    expected_hybrid_policy["absolute_power_ratio_min"]
                )
                selected_power_ratio = hybrid_power_ratio[hybrid_selected]
                if (
                    not np.isfinite(selected_power_ratio).all()
                    or np.any(selected_power_ratio < minimum_power_ratio)
                ):
                    raise ValueError(
                        "Hybrid prediction contains a selected v2.1 output "
                        "below its configured 4-30 Hz power floor."
                    )
            if np.any(hybrid_selected & ~accepted_model_output):
                raise ValueError(
                    "Hybrid prediction selects v2.1 inside an exact-copy "
                    "fallback state."
                )

            base_slow_corr = np.asarray(
                npz["hybrid_v3_slow_corr_window"], dtype=bool
            )
            final_slow_corr = np.asarray(
                npz["hybrid_final_slow_corr_window"], dtype=bool
            )
            window_shape = (len(expected_source),)
            _require_shape(
                "hybrid_v3_slow_corr_window",
                base_slow_corr,
                window_shape,
            )
            _require_shape(
                "hybrid_final_slow_corr_window",
                final_slow_corr,
                window_shape,
            )
            if np.any(final_slow_corr & ~base_slow_corr):
                raise ValueError(
                    "Hybrid prediction introduces a new SLOW_CORR window."
                )

        metadata = AutoencoderPredictionMetadata(
            prediction_file=str(path),
            checkpoint=str(_scalar(npz, "autoencoder_checkpoint")),
            model_variant=str(_scalar(npz, "autoencoder_variant")),
            model_stage=str(_scalar(npz, "autoencoder_stage")),
            policy_version=policy_version,
            ecg_min_abs_correlation=threshold,
            processed_channel_windows=int(processed.sum()),
            preserved_strict_clean_channel_windows=int(preserved.sum()),
            bypassed_marker_channel_windows=int(bypassed.sum()),
            output_safety_fallback_channel_windows=int(
                safety_fallback.sum()
            ),
            output_safety_new_flatline_channel_windows=int(
                safety_new_flatline.sum()
            ),
            output_safety_new_clipping_channel_windows=int(
                safety_new_clipping.sum()
            ),
            low_information_bypassed_channel_windows=int(
                low_information_bypassed.sum()
            ),
            selective_preservation_fallback_channel_windows=int(
                selective_fallback.sum()
            ),
            hybrid_policy_version=hybrid_policy_version,
            hybrid_selected_v21_channel_windows=int(
                hybrid_selected.sum()
            ),
        )

    # Start with the current Float64 pipeline signals.  Prediction files store
    # their source and model output as Float32, so writing copied policy windows
    # back from the NPZ would introduce a small quantisation change even though
    # they were deliberately not processed.  Only channel windows that actually
    # passed through the autoencoder may replace the current original signal.
    output1 = eeg1.copy()
    output2 = eeg2.copy()
    for window_number in range(len(denoised)):
        start = window_number * window_size
        end = start + window_size
        if accepted_model_output[window_number, 0]:
            output1[start:end] = denoised[window_number, 0]
        if accepted_model_output[window_number, 1]:
            output2[start:end] = denoised[window_number, 1]

    return output1, output2, metadata
