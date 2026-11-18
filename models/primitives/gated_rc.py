"""Standalone parallel resistor and EGT branch for charging and discharging, assuming
source-drain symmetry.
"""
import math

import torch
from torch import nn

from .egt import FengEGT


class GatedRCBranch(nn.Module):
    """Drive a grounded capacitor from an ideal target voltage through a resistor in
    parallel with an EGT. The readout draws no current.

    Enabling the transistor requires ``assume_reciprocal=True`` to accept the source-drain
    symmetry assumption. With ``transistor_enabled=False``, only the resistor carries
    current and the EGT model is skipped. Resistance is in ohms and capacitance in farads.
    Capacitance is a fixed buffer. Resistance can be learned within bounds in log space. The
    caller carries the capacitor state between updates.
    """

    def __init__(self, *, assume_reciprocal, resistance=1e6, capacitance=1e-6,
                 transistor_enabled=True, learnable_resistance=False,
                 resistance_bounds=(1e3, 1e9), instances=1):
        super().__init__()
        if type(transistor_enabled) is not bool:
            raise ValueError("transistor_enabled must be a boolean.")
        self.transistor_enabled = transistor_enabled
        if transistor_enabled and assume_reciprocal is not True:
            raise ValueError("Bidirectional branch requires explicit assume_reciprocal=True.")
        if not math.isfinite(capacitance) or capacitance <= 0:
            raise ValueError("capacitance must be finite and positive.")
        self.register_buffer("capacitance", torch.tensor(capacitance, dtype=torch.float64))
        self.learnable_resistance = learnable_resistance
        if learnable_resistance:
            if type(instances) is not int or instances < 1:
                raise ValueError("instances must be a positive integer for a learnable resistance.")
            if (not isinstance(resistance_bounds, (list, tuple)) or len(resistance_bounds) != 2
                    or any(not math.isfinite(value) or value <= 0 for value in resistance_bounds)):
                raise ValueError("resistance_bounds must contain two finite positive values.")
            lower, upper = float(resistance_bounds[0]), float(resistance_bounds[1])
            if lower >= upper or not lower <= resistance <= upper:
                raise ValueError("Resistance must lie within resistance_bounds.")
            self.resistance_bounds = (lower, upper)
            log_lower, log_upper = math.log(lower), math.log(upper)
            log_resistance = math.log(resistance)
            fraction = (log_resistance - log_lower) / (log_upper - log_lower)
            # Invert the sigmoid used by _resistance; keep boundary starts finite.
            epsilon = torch.finfo(torch.float64).eps
            fraction = min(max(fraction, epsilon), 1.0 - epsilon)
            logit = math.log(fraction) - math.log1p(-fraction)
            self.log_resistance = nn.Parameter(
                torch.full((instances,), logit, dtype=torch.float64)
            )
        else:
            if not math.isfinite(resistance) or resistance <= 0:
                raise ValueError("resistance must be finite and positive.")
            self.register_buffer("resistance", torch.tensor(resistance, dtype=torch.float64))
        self.egt = FengEGT()

    @staticmethod
    def _scale(value, state):
        value = torch.as_tensor(value, dtype=state.dtype, device=state.device)
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError("R/C scale must be finite and positive.")
        return value

    def _resistance(self, state):
        """Return positive resistances, mapping learned parameters into the log-space bounds
        with a sigmoid.
        """
        if not self.learnable_resistance:
            return self.resistance.to(state)
        lower, upper = self.resistance_bounds
        log_lower, log_upper = math.log(lower), math.log(upper)
        log_value = log_lower + (log_upper - log_lower) * torch.sigmoid(self.log_resistance)
        return torch.exp(log_value).to(state)

    def _conductance(self, state, target, gate, r_scale=1.0):
        if any(v.dtype != state.dtype or v.device != state.device for v in (target, gate)):
            raise ValueError("All voltage tensors must have the same dtype and device.")
        if state.dtype not in (torch.float32, torch.float64):
            raise TypeError("Voltage tensors must use float32 or float64.")
        passive_conductance = 1 / (self._resistance(state) * self._scale(r_scale, state))
        if not self.transistor_enabled:
            return passive_conductance
        # Re-label the lower terminal as local source: assumed reciprocity only.
        vgs = gate - torch.minimum(target, state)
        drop = (target - state).abs()
        # Query the already-reviewed small-VDS model to avoid division by zero.
        # At zero drop, its first-order value yields a finite voltage gradient.
        probe = drop.clamp_min(8 * torch.finfo(state.dtype).eps)
        return passive_conductance + self.egt(vgs, probe) / probe

    def current(self, state, target, gate, *, r_scale=1.0):
        """Return the total current into the capacitor (A). Negative current discharges it."""
        return self._conductance(state, target, gate, r_scale) * (target - state)

    def forward(self, state, target, gate, dt, substeps=1, *, r_scale=1.0, c_scale=1.0, trace=None):
        """Advance the state by ``dt`` seconds at fixed target and gate voltages and return
        the capacitor voltage (V).

        At each substep the optional trace callback receives the initial branch quantities
        and the states before and after the update. The caller stores the trace.

        The exponential Euler update freezes the secant conductance over each substep. This
        gives a first-order approximation to the nonlinear ODE. The bounded update still
        requires a step-size convergence check. The branch neither clips nor stores the
        state.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer.")
        capacitance = self.capacitance.to(state) * self._scale(c_scale, state)
        for _ in range(substeps):
            conductance = self._conductance(state, target, gate, r_scale)
            alpha = -torch.expm1(-(dt / substeps) * conductance / capacitance)
            previous = state
            state = state + alpha * (target - state)
            if trace is not None:
                trace(dict(state_before=previous, state_after=state, target=target.expand_as(state),
                           gate=gate.expand_as(state) if self.transistor_enabled else None,
                           conductance=conductance.expand_as(state),
                           current=(conductance * (target-previous)).expand_as(state),
                           tau=(capacitance / conductance).expand_as(state)))
        return state
