"""Direct marker features used by the inference safety checks.

Functions are extracted unchanged from the validated marker implementation.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch

EPSILON = 1e-10

def _bandpower(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    selection = (frequencies >= low) & (frequencies <= high)
    if selection.sum() < 2:
        return np.zeros(spectrum.shape[0], dtype=np.float64)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return trapezoid(spectrum[:, selection], frequencies[selection], axis=1)


def signal_features(
    signals: np.ndarray,
    sample_rate: float,
) -> dict[str, np.ndarray]:
    """Calculate the direct single-channel ADE marker features."""

    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[None, :]
    if signals.ndim != 2 or signals.shape[1] < 4:
        raise ValueError("signals must have shape [windows, samples].")
    if not np.isfinite(signals).all():
        raise ValueError("signals contain NaN or infinite values.")

    count, sample_count = signals.shape
    variance = np.var(signals, axis=1)
    std = np.sqrt(variance)
    minimum = np.min(signals, axis=1)
    maximum = np.max(signals, axis=1)
    ptp = maximum - minimum
    differences = np.diff(signals, axis=1)
    max_jump = np.max(np.abs(differences), axis=1)
    line_length = np.sum(np.abs(differences), axis=1)
    max_jump_norm = max_jump / (std + EPSILON)
    line_length_norm = line_length / (sample_count * (std + EPSILON))

    x = np.arange(sample_count, dtype=np.float64)
    x_centered = x - x.mean()
    slopes = (signals @ x_centered) / np.sum(np.square(x_centered))
    trend_slope_norm = np.abs(slopes) * sample_count / (std + EPSILON)

    tolerance = 0.005 * ptp
    clipping_ratio = np.ones(count, dtype=np.float64)
    nonconstant = ptp > EPSILON
    clipping_ratio[nonconstant] = np.mean(
        (
            np.abs(signals[nonconstant] - minimum[nonconstant, None])
            <= tolerance[nonconstant, None]
        )
        | (
            np.abs(signals[nonconstant] - maximum[nonconstant, None])
            <= tolerance[nonconstant, None]
        ),
        axis=1,
    )

    frequencies, spectrum = welch(
        signals,
        fs=float(sample_rate),
        nperseg=min(256, sample_count // 2),
        axis=1,
    )
    total_30 = _bandpower(frequencies, spectrum, 0.5, 30.0)
    total_45 = _bandpower(frequencies, spectrum, 0.5, 45.0)
    total_55 = _bandpower(frequencies, spectrum, 0.5, 55.0)

    return {
        "variance": variance,
        "std": std,
        "ptp": ptp,
        "max_jump_norm": max_jump_norm,
        "line_length_norm": line_length_norm,
        "trend_slope_norm": trend_slope_norm,
        "clipping_ratio": clipping_ratio,
        "delta_rel": _bandpower(frequencies, spectrum, 0.5, 4.0)
        / (total_30 + EPSILON),
        "beta_rel": _bandpower(frequencies, spectrum, 13.0, 30.0)
        / (total_30 + EPSILON),
        "hf_rel": _bandpower(frequencies, spectrum, 30.0, 45.0)
        / (total_45 + EPSILON),
        "line_50_rel": _bandpower(frequencies, spectrum, 49.0, 51.0)
        / (total_55 + EPSILON),
    }


def derive_recording_thresholds(
    eeg: np.ndarray,
    sample_rate: float,
) -> dict[str, float]:
    """Reconstruct the recording-adaptive direct ADE thresholds."""

    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 3 or eeg.shape[1] != 2:
        raise ValueError("eeg must have shape [windows, 2, samples].")
    features = signal_features(eeg.reshape(-1, eeg.shape[-1]), sample_rate)
    return {
        "threshold_var_emg": float(np.quantile(features["variance"], 0.75)),
        "threshold_amp_burst": float(np.quantile(features["ptp"], 0.95)),
        "threshold_jump": float(
            np.quantile(features["max_jump_norm"], 0.95)
        ),
        "threshold_flatline_std": float(np.quantile(features["std"], 0.01)),
        "threshold_flatline_ptp": float(np.quantile(features["ptp"], 0.01)),
        "threshold_hf_emg_line": float(
            np.quantile(features["line_length_norm"], 0.75)
        ),
        "threshold_low_drift_slope": float(
            np.quantile(features["trend_slope_norm"], 0.90)
        ),
        "threshold_beta_emg_rel": 0.25,
        "threshold_hf_emg_rel": 0.15,
        "threshold_low_drift_delta_rel": 0.80,
        "threshold_clipping_ratio": 0.02,
    }
