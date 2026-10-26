"""One-dimensional denoising autoencoders for single-channel EEG windows."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class ConvBlock(nn.Module):
    """Convolution, group normalisation and a smooth activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 7,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        group_count = min(8, out_channels)
        while out_channels % group_count != 0:
            group_count -= 1
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(group_count, out_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SignalEncoder(nn.Module):
    """Compact fully convolutional encoder shared by all window lengths."""

    def __init__(self, in_channels: int, base: int, latent: int) -> None:
        super().__init__()
        self.level1 = ConvBlock(in_channels, base, kernel_size=9)
        self.level2 = ConvBlock(base, base * 2, stride=2)
        self.level3 = ConvBlock(base * 2, latent, stride=2)
        self.bottleneck = ConvBlock(latent, latent, kernel_size=5)

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        level1 = self.level1(values)
        level2 = self.level2(level1)
        latent = self.bottleneck(self.level3(level2))
        return latent, (level1.shape[-1], level2.shape[-1])


class DenoisingAutoencoder1D(nn.Module):
    """Denoise one EEG channel, optionally conditioned on markers and ECG.

    The same network is applied independently to EEG1 and EEG2.  The
    conditioned variant fuses two additional sources at the bottleneck:

    * a reference waveform (currently ECG when available), and
    * a fixed-order vector of rule/synthetic artifact markers.

    The unconditioned variant receives the same method arguments so the
    training and inference code can remain identical, but ignores them.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        latent = config.latent_channels

        self.encoder = SignalEncoder(config.input_channels, base, latent)

        if config.conditioned:
            self.reference_encoder = SignalEncoder(1, base, latent)
            self.marker_projection = nn.Sequential(
                nn.Linear(config.marker_count, latent),
                nn.GELU(),
                nn.Linear(latent, latent),
            )
            # Columns for marker types that are not represented by the current
            # synthetic training generator stay neutral instead of retaining
            # arbitrary random effects. Active marker columns can still learn
            # because the second linear layer is conventionally initialised.
            nn.init.zeros_(self.marker_projection[0].weight)
            nn.init.zeros_(self.marker_projection[0].bias)
            nn.init.zeros_(self.marker_projection[2].bias)
            # Marker columns that are never represented by a defensible
            # synthetic corruption must initially have exactly no effect.
            # Active columns can still learn from their training examples.
            nn.init.zeros_(self.marker_projection[0].weight)
            nn.init.zeros_(self.marker_projection[0].bias)
            nn.init.xavier_uniform_(self.marker_projection[2].weight)
            nn.init.zeros_(self.marker_projection[2].bias)
            self.condition_gate = nn.Sequential(
                nn.Conv1d(latent * 3, latent, kernel_size=1),
                nn.Sigmoid(),
            )
            self.condition_candidate = ConvBlock(
                latent * 3,
                latent,
                kernel_size=1,
            )

        self.decode_level2 = ConvBlock(latent, base * 2, kernel_size=5)
        self.decode_level1 = ConvBlock(base * 2, base, kernel_size=7)
        self.output_layer = nn.Conv1d(
            base,
            config.output_channels,
            kernel_size=9,
            padding=4,
        )
        if config.residual_learning:
            # Start as an identity mapping and learn the artifact residual.
            # This makes preservation of already clean windows the neutral
            # initial behaviour.
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def _validate_inputs(
        self,
        values: torch.Tensor,
        reference: torch.Tensor | None,
        markers: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError(
                "values must have shape [batch, 1, samples], got "
                f"{tuple(values.shape)}."
            )

        batch_size, _, sample_count = values.shape
        if reference is None:
            reference = torch.zeros_like(values)
        if reference.shape != (batch_size, 1, sample_count):
            raise ValueError(
                "reference must match values shape, got "
                f"{tuple(reference.shape)} and {tuple(values.shape)}."
            )

        if markers is None:
            markers = values.new_zeros((batch_size, self.config.marker_count))
        if markers.shape != (batch_size, self.config.marker_count):
            raise ValueError(
                "markers must have shape "
                f"[{batch_size}, {self.config.marker_count}], got "
                f"{tuple(markers.shape)}."
            )
        return reference, markers

    def forward(
        self,
        values: torch.Tensor,
        reference: torch.Tensor | None = None,
        markers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference, markers = self._validate_inputs(values, reference, markers)
        latent, (level1_length, level2_length) = self.encoder(values)

        if self.config.conditioned:
            reference_latent, _ = self.reference_encoder(reference)
            if reference_latent.shape[-1] != latent.shape[-1]:
                reference_latent = F.interpolate(
                    reference_latent,
                    size=latent.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            reference_present = (
                reference.abs().amax(dim=(1, 2), keepdim=True) > 0
            ).to(reference_latent.dtype)
            reference_latent = reference_latent * reference_present
            marker_latent = self.marker_projection(markers).unsqueeze(-1)
            marker_latent = marker_latent.expand(-1, -1, latent.shape[-1])
            condition_values = torch.cat(
                [latent, reference_latent, marker_latent],
                dim=1,
            )
            gate = self.condition_gate(condition_values)
            candidate = self.condition_candidate(condition_values)
            latent = latent + gate * candidate

        decoded = F.interpolate(
            latent,
            size=level2_length,
            mode="linear",
            align_corners=False,
        )
        decoded = self.decode_level2(decoded)
        decoded = F.interpolate(
            decoded,
            size=level1_length,
            mode="linear",
            align_corners=False,
        )
        decoded = self.decode_level1(decoded)
        decoded = self.output_layer(decoded)
        if self.config.residual_learning:
            return values - decoded
        return decoded


def build_model(config: ModelConfig) -> DenoisingAutoencoder1D:
    return DenoisingAutoencoder1D(config)


def load_shared_weights(
    model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Warm-start all identically named and shaped tensors.

    This supports deliberate transfer between unconditioned and conditioned
    variants while reporting exactly which tensors could not be reused.
    """

    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    incompatible_source = sorted(set(source_state).difference(compatible))
    missing_target = sorted(set(target_state).difference(compatible))
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "not_loaded_from_source": incompatible_source,
        "new_or_missing_in_target": missing_target,
    }
