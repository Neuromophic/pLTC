"""Resistive crossbar primitive."""

import re

import torch
from torch import nn

from .circuits import ForwardResult, power_requested
from .variation import UniformMultiplicativeVariation


class Crossbar(nn.Module):
    """Printable resistive crossbar with inverters for negative weights."""

    device_inventory = {"resistor": 1}
    _SPICE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    def __init__(
        self, n_in, n_out, args, activation_operating_input=None, variation=None
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.args = args
        self.variation = variation or UniformMultiplicativeVariation()
        theta = torch.rand((n_in + 2, n_out), dtype=torch.float32) / 100.0 + args.gmin
        theta[-1, :] += args.gmax
        if activation_operating_input is not None:
            reference_input = torch.as_tensor(
                activation_operating_input, dtype=theta.dtype
            )
            mapped_input = (reference_input if getattr(args, "temporal_protocol", None) == "learnable_filters"
                            else torch.tanh(reference_input))
            theta[-2, :] = (
                mapped_input
                / (1.0 - mapped_input + torch.finfo(theta.dtype).tiny)
                * theta[:-2, :].sum(dim=0).add(theta[-1, :])
            )
        if not torch.isfinite(theta).all():
            raise FloatingPointError(
                "Crossbar initialization produced non-finite conductance values."
            )
        self.theta_ = nn.Parameter(theta)
        self.register_buffer("pruning_mask", torch.ones_like(theta))
        self.stage = "train"

    @property
    def theta(self):
        if getattr(self.args, "temporal_protocol", None) == "learnable_filters":
            # Match the public pLayer projection and identity gradient. Using data
            # here deliberately mirrors its projected parameter, not a clamp gradient.
            self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
            thresholded = torch.where(self.theta_.abs() < self.args.gmin, 0.0, self.theta_)
            return (thresholded.detach() + self.theta_ - self.theta_.detach()) * self.pruning_mask
        bounded = self.theta_.clamp(-self.args.gmax, self.args.gmax)
        thresholded = torch.where(
            bounded.abs() < self.args.gmin,
            torch.zeros_like(bounded),
            bounded,
        )
        masked_bounded = bounded * self.pruning_mask
        masked_thresholded = thresholded * self.pruning_mask
        return masked_thresholded.detach() + masked_bounded - masked_bounded.detach()

    def prune_positions(self, positions):
        """Permanently prune the specified connections and return the number newly pruned."""
        positions = torch.as_tensor(
            positions, dtype=torch.bool, device=self.pruning_mask.device
        )
        if positions.shape != self.pruning_mask.shape:
            raise ValueError(
                f"Pruning positions must have shape {tuple(self.pruning_mask.shape)}."
            )
        newly_pruned = positions & self.pruning_mask.bool()
        with torch.no_grad():
            self.pruning_mask.masked_fill_(newly_pruned, 0.0)
        return int(newly_pruned.sum().item())

    @property
    def g_tilde(self):
        """Compute physical crossbar conductances from the thresholded effective conductances."""
        conductance = self.theta.abs()
        active = conductance != 0
        minimum = (
            torch.where(
                active,
                conductance,
                torch.full_like(conductance, torch.inf),
            )
            .min(dim=0, keepdim=True)
            .values
        )
        minimum = torch.where(
            torch.isfinite(minimum), minimum, torch.ones_like(minimum)
        )
        return conductance * (
            self.args.pgmin / (minimum + torch.finfo(minimum.dtype).tiny)
        )

    def _sampling(self):
        if self.stage in {"train", "valid"}:
            return self.args.N_train, self.args.e_train
        if self.stage == "test":
            return self.args.N_test, self.args.e_test
        raise ValueError(f"Unsupported crossbar stage: {self.stage}.")

    def _sample_conductance_variation(self):
        sample_count, epsilon = self._sampling()
        return self.variation.sample(self.theta, sample_count, epsilon)

    def _weights_from_variation(self, variation):
        if getattr(self.args, "temporal_protocol", None) == "learnable_filters":
            magnitude = self.theta.abs().unsqueeze(0) * variation.abs()
            total = magnitude.sum(dim=1, keepdim=True)
            return magnitude / torch.where(total > 0, total, torch.ones_like(total))
        # Keep the hard physical conductance in forward, but take the magnitude
        # derivative before thresholding: abs(0) would otherwise cancel the STE.
        hard = self.theta.detach().abs()
        soft = (self.theta_.clamp(-self.args.gmax, self.args.gmax) * self.pruning_mask).abs()
        active_column = hard.sum(dim=0, keepdim=True) > 0
        magnitude = hard + (soft - soft.detach()) * active_column
        varied = magnitude.unsqueeze(0) * variation.abs()
        total = varied.sum(dim=1, keepdim=True)
        # A fully disconnected column stays zero, with no undefined 0/0 gradient.
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        return varied / denominator

    def _active_input_inverter_mask(self):
        """Return the mask of inverters required for signal inputs from the preceding layer."""
        inverter_theta = self.theta[: self.n_in]
        hard = (inverter_theta < 0).any(dim=1).to(dtype=self.theta.dtype)
        negative = (inverter_theta < 0).to(dtype=self.theta.dtype)
        relaxed = ((1.0 - torch.sigmoid(inverter_theta)) * negative).max(dim=1).values
        return hard.detach() + relaxed - relaxed.detach()

    @classmethod
    def _validate_spice_identifier(cls, value, field):
        """Validate a SPICE identifier for the crossbar subcircuit or one of its dependencies."""
        if not isinstance(value, str) or cls._SPICE_IDENTIFIER.fullmatch(value) is None:
            raise ValueError(f"Invalid SPICE {field}: {value!r}.")
        return value

    @staticmethod
    def _spice_resistance(conductance):
        """Convert a positive conductance to a SPICE resistance literal."""
        if not torch.isfinite(conductance) or conductance <= 0:
            raise ValueError("Crossbar conductance must be finite and positive.")
        resistance = 1.0 / float(conductance)
        if resistance >= 1e6:
            return f"{resistance / 1e6:.9g}meg"
        if resistance >= 1e3:
            return f"{resistance / 1e3:.9g}k"
        return f"{resistance:.9g}"

    def spice_subcircuit(self, name="crossbar", inverter_subcircuit="inverter"):
        """Generate a SPICE subcircuit for the nominal crossbar.

        Each signal row has a ``VIN_i`` port. A row with active negative connections uses
        one ``inverter_subcircuit`` instance and routes those connections to ``VIN_i_NEG``.
        The bias row uses ``VBIAS_POS`` and ``VBIAS_NEG``. The reference row connects to
        ``GND``, so each high-impedance output node gives the conductance-weighted average
        for its column.
        """
        name = self._validate_spice_identifier(name, "subcircuit name")
        inverter_subcircuit = self._validate_spice_identifier(
            inverter_subcircuit, "inverter subcircuit name"
        )
        theta = self.theta.detach().cpu()
        conductance = self.g_tilde.detach().cpu()
        if not torch.isfinite(theta).all() or not torch.isfinite(conductance).all():
            raise FloatingPointError("Crossbar SPICE export received non-finite values.")

        inputs = [f"VIN_{index}" for index in range(self.n_in)]
        outputs = [f"VOUT_{index}" for index in range(self.n_out)]
        ports = inputs + ["VBIAS_POS", "VBIAS_NEG"] + outputs + ["VDD", "GND"]
        lines = [
            f"* Signed resistive crossbar, {self.n_in} inputs x {self.n_out} outputs",
            f".SUBCKT {name} {' '.join(ports)}",
        ]

        active_negative_inputs = (theta[: self.n_in] < 0).any(dim=1)
        for row, required in enumerate(active_negative_inputs.tolist()):
            if required:
                lines.append(
                    f"XINV_{row} VIN_{row} VIN_{row}_NEG VDD GND {inverter_subcircuit}"
                )

        for row in range(self.n_in + 2):
            for column in range(self.n_out):
                signed_weight = theta[row, column]
                if signed_weight == 0:
                    continue
                if row < self.n_in:
                    source = (
                        f"VIN_{row}" if signed_weight > 0 else f"VIN_{row}_NEG"
                    )
                elif row == self.n_in:
                    source = "VBIAS_POS" if signed_weight > 0 else "VBIAS_NEG"
                else:
                    source = "GND"
                resistance = self._spice_resistance(conductance[row, column])
                lines.append(f"R_{row}_{column} {source} VOUT_{column} {resistance}")

        lines.append(f".ENDS {name}")
        return "\n".join(lines) + "\n"

    def has_negative_bias_connection(self):
        """Return whether the crossbar needs the inverted global bias voltage."""
        return bool((self.theta.detach()[-2] < 0).any().item())

    def active_activation_mask(self):
        """Return the hidden activation mask with a straight-through gradient.

        The forward pass enables an activation only if its output column has a surviving
        signal input. Device counts, evaluation, and deployment reports use this binary
        mask. During backpropagation a smooth function of the largest absolute conductance
        in each column supplies the gradient, so the activation power penalty can drive the
        column toward the pruning threshold. The input inverter mask uses the same
        straight-through estimator.
        """
        activation_theta = self.theta[: self.n_in]
        connected = (activation_theta != 0).to(dtype=self.theta.dtype)
        hard = connected.any(dim=0).to(dtype=self.theta.dtype)
        relaxed = (torch.sigmoid(activation_theta.abs()) * connected).max(dim=0).values
        return hard.detach() + relaxed - relaxed.detach()

    def _mac_power(
        self, extended_input, negative_input, output, positive, negative, conductance
    ):
        cell_input = (
            extended_input.unsqueeze(-1) * positive
            + negative_input.unsqueeze(-1) * negative
        )
        voltage = cell_input - output.unsqueeze(-2)
        denominator = (
            extended_input.shape[0] * extended_input.shape[1]
            + torch.finfo(extended_input.dtype).tiny
        )
        return (conductance.unsqueeze(1) * voltage.square()).sum() / denominator

    def forward(self, signal, inverter, inverted_bias):
        if signal.ndim != 3 or signal.shape[-1] != self.n_in:
            raise ValueError(
                f"Crossbar signal must have shape [sample, batch, {self.n_in}]."
            )
        if inverted_bias.shape != (*signal.shape[:2], 1):
            raise ValueError("inverted_bias must have shape [sample, batch, 1].")
        extended = torch.cat(
            (
                signal,
                torch.ones_like(signal[..., :1]),
                torch.zeros_like(signal[..., :1]),
            ),
            dim=-1,
        )
        # A thresholded weight has zero forward magnitude; retain its latent sign
        # to route its surrogate derivative through the correct physical branch.
        sign_source = self.theta if getattr(self.args, "temporal_protocol", None) == "learnable_filters" else self.theta_
        positive = (sign_source >= 0).to(dtype=signal.dtype)
        negative = 1.0 - positive
        inverter_result = inverter(
            signal, active_device_mask=self._active_input_inverter_mask()
        )
        negative_input = torch.cat(
            (inverter_result.output, inverted_bias, torch.zeros_like(inverted_bias)),
            dim=-1,
        )
        variation = self._sample_conductance_variation()
        weights = self._weights_from_variation(variation)
        output = torch.matmul(extended, weights * positive) + torch.matmul(
            negative_input, weights * negative
        )
        if not torch.isfinite(output).all():
            raise FloatingPointError("Crossbar output contains non-finite values.")
        if not power_requested(self.args):
            # MAC power and the inverter power estimate are unused without a power
            # objective; skipping them keeps the output bit-identical.
            return ForwardResult(output=output, power=output.new_zeros(()))
        conductance = self.g_tilde.unsqueeze(0) * variation.abs()
        mac_power = self._mac_power(
            extended, negative_input, output, positive, negative, conductance
        )
        power = mac_power + inverter_result.power
        if not torch.isfinite(power):
            raise FloatingPointError("Crossbar power is non-finite.")
        if power < 0:
            raise FloatingPointError("Crossbar power must be non-negative.")
        return ForwardResult(output=output, power=power)

    def device_count(self):
        theta = self.theta.detach()
        return {
            "crossbar_instances": int((theta != 0).sum().item()),
            "inverter_instances": int((theta[: self.n_in] < 0).any(dim=1).sum().item()),
        }

    def initialize_operating_sensitivity(self, input_width, input_share, state_share,
                                         operating_input):
        """Initialize the input and feedback rows to specified fractions of each normalized
        column.

        In the default pLTC gate crossbar, signal rows account for about 0.4% of each
        column's weight, so the input has little effect on gate voltage. The first
        ``input_width`` rows carry external inputs and receive a combined fraction
        ``input_share``. The remaining signal rows carry state feedback and receive
        ``state_share``. With zero input the output stays near ``operating_input``. The
        reserve row is fixed at ``gmax`` to leave headroom for normalization.

        This initialization is for positive-conductance layers and requires
        ``operating_input`` in (0, 1).
        """
        if input_width < 0 or input_width > self.n_in:
            raise ValueError("input_width must fall inside the signal rows.")
        for name, value in (("input_share", input_share), ("state_share", state_share)):
            if not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0,1).")
        if not isinstance(operating_input, (int, float)) or not 0 < operating_input < 1:
            raise ValueError("operating_input must be in (0,1) for this reweighting.")
        denominator = 1.0 - operating_input - input_share - state_share
        if denominator <= 0:
            raise ValueError("input_share + state_share + operating_input must be below 1.")
        reserve = float(self.args.gmax)
        total = reserve / denominator
        state_rows = self.n_in - input_width
        with torch.no_grad():
            self.theta_.data[: self.n_in] = 0.0
            if input_width:
                self.theta_.data[: input_width] = input_share * total / input_width
            if state_rows:
                self.theta_.data[input_width: self.n_in] = state_share * total / state_rows
            self.theta_.data[self.n_in] = operating_input * total
            self.theta_.data[self.n_in + 1] = reserve
        return self

    def SetParameter(self, name, value):
        if name == "args":
            self.args = value
        elif name == "stage":
            if value not in {"train", "valid", "test"}:
                raise ValueError(f"Unsupported crossbar stage: {value}.")
            self.stage = value
