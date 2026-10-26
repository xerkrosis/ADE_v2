"""Model architecture configuration used to reconstruct supplied checkpoints."""
from __future__ import annotations
from dataclasses import asdict, dataclass

MODEL_VARIANTS = ("unconditioned", "conditioned")
CONDITION_MARKERS = (
    "CLEAN",
    "EOG",
    "EMG",
    "ECG",
    "SLOW_CORR",
    "BAD_AUTO",
    "WAVELET_HF",
    "WAVELET_TRANSIENT",
    "AMP_BURST",
    "JUMP",
    "FLATLINE",
    "HF_EMG",
    "LOW_DRIFT",
    "CLIPPING",
    "LINE_NOISE",
)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters saved with every checkpoint."""

    variant: str = "unconditioned"
    input_channels: int = 1
    output_channels: int = 1
    base_channels: int = 32
    latent_channels: int = 128
    residual_learning: bool = True
    marker_names: tuple[str, ...] = CONDITION_MARKERS

    def __post_init__(self) -> None:
        if self.variant not in MODEL_VARIANTS:
            raise ValueError(
                f"variant must be one of {MODEL_VARIANTS}, got {self.variant!r}."
            )
        if self.input_channels != 1 or self.output_channels != 1:
            raise ValueError(
                "The model is deliberately single-channel so the same "
                "backbone can be pretrained with EEGdenoiseNet and applied "
                "independently to DOSE-I EEG1 and EEG2."
            )
        if self.base_channels < 1 or self.latent_channels < 1:
            raise ValueError("base_channels and latent_channels must be positive.")
        if len(set(self.marker_names)) != len(self.marker_names):
            raise ValueError("marker_names must not contain duplicates.")

    @property
    def conditioned(self) -> bool:
        return self.variant == "conditioned"

    @property
    def marker_count(self) -> int:
        return len(self.marker_names)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["marker_names"] = list(self.marker_names)
        return result

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        values = dict(values)
        if "marker_names" in values:
            values["marker_names"] = tuple(values["marker_names"])
        return cls(**values)
