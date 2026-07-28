"""Differentiable, device-native discrete wavelet transform.

:class:`WaveletDecompose1d` reproduces :func:`pywt.wavedec` with ``mode="symmetric"`` to
floating-point tolerance, but as a cascade of fixed grouped convolutions. That matters for
three reasons the research prototype hit:

* it runs wherever the tensor lives, so a batch is no longer copied GPU -> CPU -> NumPy ->
  GPU on every forward pass;
* it is differentiable, so the decomposition can sit inside a model without blocking
  gradients (the filters themselves stay fixed, registered as buffers, not parameters);
* it is vectorised over batch and channel instead of a Python double loop.

Equivalence to PyWavelets is asserted in ``tests/unit/test_dwt_equivalence.py`` for several
wavelets, levels and signal lengths.

Implementation note: PyWavelets pads symmetrically by ``dec_len - 1`` samples, correlates
with the reversed decomposition filters, then keeps every second output starting at index
1. ``torch.nn.functional.pad`` has no ``"symmetric"`` mode (its ``"reflect"`` does not
repeat the edge sample), so the padding is done by index gather, which is differentiable
and works on any device.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pywt
import torch
from torch import nn

from bruxism.preprocessing.wavelets import WaveletConfig, band_index

__all__ = ["WaveletDecompose1d", "symmetric_pad"]


def symmetric_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Symmetric ("whole-sample") padding of the last axis by ``pad`` samples per side.

    Matches ``numpy.pad(..., mode="symmetric")``: the edge sample is repeated, unlike
    ``torch.nn.functional.pad(mode="reflect")`` which skips it.
    """
    if pad <= 0:
        return x
    n = x.shape[-1]
    if n == 0:
        raise ValueError("cannot symmetric-pad a zero-length signal")
    # Build the gather index by folding a ramp into [0, n-1] with period 2n.
    positions = torch.arange(-pad, n + pad, device=x.device)
    folded = torch.remainder(positions, 2 * n)
    index = torch.where(folded < n, folded, 2 * n - 1 - folded)
    return x.index_select(-1, index)


class WaveletDecompose1d(nn.Module):
    """Multi-level DWT returning the bands named in a :class:`WaveletConfig`.

    Parameters
    ----------
    config
        Wavelet, level and the named bands to return.
    n_channels
        Channel count of the input, used to size the grouped convolutions.

    Shapes
    ------
    input
        ``(batch, n_channels, n_samples)``
    output
        ``dict[str, Tensor]`` with one entry per configured band, each
        ``(batch, n_channels, n_coefficients_for_that_band)``.
    """

    def __init__(self, config: WaveletConfig, n_channels: int):
        super().__init__()
        self.config = config
        self.n_channels = int(n_channels)

        wavelet = pywt.Wavelet(config.wavelet)
        self.filter_length = wavelet.dec_len
        # Correlation with the reversed analysis filters == PyWavelets' convolution.
        # Stored at full precision; forward() casts to the input dtype so a float64 model
        # reproduces PyWavelets exactly rather than to float32 tolerance.
        dec_lo = np.asarray(wavelet.dec_lo, dtype=np.float64)[::-1].copy()
        dec_hi = np.asarray(wavelet.dec_hi, dtype=np.float64)[::-1].copy()
        # (2 * n_channels, 1, filter_length): low-pass rows then high-pass rows.
        stacked = np.concatenate(
            [
                np.tile(dec_lo, (self.n_channels, 1)),
                np.tile(dec_hi, (self.n_channels, 1)),
            ],
            axis=0,
        )[:, None, :]
        self.register_buffer("weight", torch.from_numpy(stacked), persistent=False)

    def _single_level(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One decomposition step: returns ``(approximation, detail)``."""
        pad = self.filter_length - 1
        padded = symmetric_pad(x, pad)
        # Grouped conv over channels; the low/high halves are stacked along the output dim.
        doubled = padded.repeat(1, 2, 1)
        weight = cast(torch.Tensor, self.weight).to(dtype=x.dtype)
        out = torch.nn.functional.conv1d(doubled, weight, groups=2 * self.n_channels)
        out = out[..., 1::2]
        return out[:, : self.n_channels], out[:, self.n_channels :]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, channels, time), got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {x.shape[1]}; the decomposition "
                f"filters are sized at construction time"
            )
        coefficients: list[torch.Tensor] = [torch.empty(0)] * (self.config.level + 1)
        current = x
        for level in range(1, self.config.level + 1):
            current, detail = self._single_level(current)
            # pywt order is [cA_L, cD_L, ..., cD_1]; detail at step k is D_k, whose slot is
            # level - k + 1.
            coefficients[self.config.level - level + 1] = detail
        coefficients[0] = current
        return {
            band: coefficients[band_index(band, self.config.level)] for band in self.config.bands
        }

    def output_lengths(self, n_samples: int) -> dict[str, int]:
        """Coefficient length per configured band for a given input length."""
        return self.config.coefficient_lengths(n_samples)

    def extra_repr(self) -> str:
        return (
            f"wavelet={self.config.wavelet}, level={self.config.level}, "
            f"bands={list(self.config.bands)}, n_channels={self.n_channels}"
        )
