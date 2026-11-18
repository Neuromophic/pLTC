"""Standalone quasi-static EGT current-voltage model.

Feng (2021), DOI 10.5445/IR/1000137975, gives the current equations on p. 33
(Eqs. 2.26–2.30), the channel geometry (W = 100 µm, L = 60 µm) on p. 84, and
the matching inverter parameters on p. 86.

Voltages are source-referenced with VDS >= 0. Source-drain interchange, external RC and
network dynamics, and manufacturing variation are outside this device model.
"""
import math

import torch
from torch import nn


class FengEGT(nn.Module):
    """Return drain-to-source current (A) from VGS and VDS tensors (V).

    The voltages share a dtype and device and support PyTorch broadcasting with float32 or
    float64. Current is differentiable with respect to the voltages. The fitted parameters
    are fixed buffers. The model assumes a default temperature of 298 K.
    """

    def __init__(self, temperature=298.0):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")
        # Vth, S, gamma, f1, f2, f3, f4, W, L, T; SI units.
        self.register_buffer("fit_parameters", torch.tensor(
            [0.13, 16.45, 1.57, 7.6e-4, 0.78, 1.46, 0.02,
             100e-6, 60e-6, temperature], dtype=torch.float64))

    def forward(self, vgs, vds):
        if vgs.dtype not in (torch.float32, torch.float64):
            raise TypeError("Voltage tensors must use float32 or float64.")
        if vds.dtype != vgs.dtype or vds.device != vgs.device:
            raise ValueError("VGS and VDS must have the same dtype and device.")
        if not torch.isfinite(vgs).all() or not torch.isfinite(vds).all():
            raise ValueError("Voltages must be finite.")
        if (vds < 0).any():
            raise ValueError("This source-referenced model requires VDS >= 0.")
        vth, slope, gamma, f1, f2, f3, f4, width, length, temp = (
            self.fit_parameters.to(vgs).unbind())
        thermal = (1.380649e-23 / 1.602176634e-19) * temp
        n = 1 / (slope * thermal * math.log(10.0))
        i0 = 2 * n * f1 * (width / length) * thermal.square()
        vp = (vgs - vth + f4 * vds) * f2 / (n * thermal)
        # Source reference: vs=0, vd=VDS/thermal. Avoid direct exp overflow.
        z = vp / 2
        lf = torch.logaddexp(f3.log(), z)
        lr = torch.logaddexp(f3.log(), z - vds / (2 * thermal))
        current = i0 * lf.pow(gamma) * (-torch.expm1(gamma * (lr.log() - lf.log())))
        # Analytic first-order limit at VDS=0 avoids cancellation and preserves
        # a finite dIDS/dVDS at zero. This is a numerical approximation only.
        g_zero = i0 * gamma * lf.pow(gamma - 1) * torch.sigmoid(z - f3.log()) / (2 * thermal)
        small = vds <= 8 * torch.finfo(vgs.dtype).eps
        return torch.where(small, g_zero * vds, current)
