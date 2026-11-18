"""pNN layers built from a resistive crossbar and an optional activation circuit."""

from dataclasses import dataclass

import torch
from torch import nn

from .primitives.circuits import ForwardResult, NonlinearCircuit
from .primitives.gated_rc import GatedRCBranch
from .primitives.variation import SequenceVariation
from .primitives.crossbar import Crossbar


class pLayer(nn.Module):
    """Crossbar layer with an optional activation circuit. The output layer can use the
    crossbar output directly.
    """

    def __init__(
        self, n_in, n_out, args, has_activation, activation_operating_input, variation
    ):
        super().__init__()
        self.crossbar = Crossbar(
            n_in, n_out, args, activation_operating_input, variation
        )
        self.has_activation = has_activation

    def forward(self, signal, inverter, inverted_bias, activation=None):
        result = self.crossbar(signal, inverter, inverted_bias)
        if activation is None:
            return result
        activation_result = activation(
            result.output,
            active_device_mask=self.crossbar.active_activation_mask(),
        )
        return ForwardResult(
            output=activation_result.output,
            power=result.power + activation_result.power,
        )

    def SetParameter(self, name, value):
        self.crossbar.SetParameter(name, value)


@dataclass
class StateCellResult:
    """``state`` holds the capacitor voltage and ``output`` holds the activation circuit's
    readout.
    """
    state: torch.Tensor
    output: torch.Tensor


class GatedStateCell(nn.Module):
    """Combine a state branch and readout circuit with externally supplied target and gate
    voltages.

    ``assume_ideal_readout=True`` neglects input leakage, input capacitance, and feedback
    coupling at the readout. For nonzero readout variation, run the cell in the temporal
    network's ``sequence()`` context.
    """

    def __init__(self, branch, readout, *, assume_ideal_readout):
        super().__init__()
        if assume_ideal_readout is not True:
            raise ValueError("Require explicit assume_ideal_readout=True for unloaded sensing.")
        if not isinstance(branch, GatedRCBranch) or not isinstance(readout, NonlinearCircuit):
            raise TypeError("Pass an existing GatedRCBranch and NonlinearCircuit instance.")
        self.branch = branch
        self.readout = readout

    def forward(self, state, target, gate, dt, substeps=1, *, r_scale=1.0, c_scale=1.0, trace=None):
        """Update and read out ``state``, a voltage tensor shaped [sample, batch, instance].

        The target and gate voltages broadcast to the state shape and stay fixed during this
        call. Pass ``result.state`` to the next step. Any feedback from the activation
        output to the target or gate is external to the cell.
        """
        if state.ndim != 3 or min(state.shape) < 1:
            raise ValueError("state must be nonempty [sample, batch, instance].")
        if torch.broadcast_shapes(state.shape, target.shape, gate.shape) != state.shape:
            raise ValueError("target/gate must broadcast to the existing state shape.")
        samples, epsilon = self.readout.sampling
        if epsilon != 0 and not isinstance(self.readout.variation, SequenceVariation):
            raise ValueError("Use nominal readout or a sequence() context for nonzero variation.")
        if state.shape[0] != samples:
            raise ValueError("State sample count must match the readout stage configuration.")
        new_state = self.branch(state, target, gate, dt, substeps, r_scale=r_scale, c_scale=c_scale, trace=trace)
        readout = self.readout(new_state)
        # The primitive retains its own power API. No total cell power is claimed.
        return StateCellResult(state=new_state, output=readout.output)
