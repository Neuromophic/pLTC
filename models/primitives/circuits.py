"""Nonlinear circuit primitives and their registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import re
from typing import ClassVar

import torch
from torch import nn

from .variation import UniformMultiplicativeVariation


@dataclass
class ForwardResult:
    """Output and power from one physical forward pass."""

    output: torch.Tensor
    power: torch.Tensor


def power_requested(args):
    """Return whether this training or evaluation pass needs circuit power estimates."""
    return getattr(args, "powerestimator", "none") != "none"


class NonlinearCircuit(nn.Module, ABC):
    """Base class for nonlinear circuits.

    Subclasses specify the initial device parameters, surrogate model assets, and transfer
    function. Within a forward pass, each physical instance has its own manufacturing
    variation sample. Output and power use the same sample.
    """

    initial_rt: ClassVar[tuple[float, ...]]
    device_inventory: ClassVar[dict[str, int]]
    model_asset: ClassVar[str]
    power_asset: ClassVar[str]
    spice_port_roles: ClassVar[tuple[str, ...]]

    _SPICE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    @classmethod
    def physical_device_count(cls) -> int:
        """Return the number of devices in one physical circuit instance."""
        return sum(cls.device_inventory.values())

    def __init__(self, args, parameter_sets=1, variation=None):
        super().__init__()
        if parameter_sets < 1:
            raise ValueError("parameter_sets must be positive.")
        self.args = args
        self.stage = "train"
        self.parameter_sets = parameter_sets
        self.variation = variation or UniformMultiplicativeVariation()
        initial = torch.tensor(self.initial_rt, dtype=torch.float32)
        self.rt_ = nn.Parameter(initial.repeat(parameter_sets, 1))
        self._load_estimators()

    @property
    def device(self):
        return self.rt_.device

    @property
    def sampling(self):
        if self.stage in {"train", "valid"}:
            return self.args.N_train, self.args.e_train
        if self.stage == "test":
            return self.args.N_test, self.args.e_test
        raise ValueError(f"Unsupported circuit stage: {self.stage}.")

    def _load_estimators(self):
        model_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        asset_dir = os.path.join(model_directory, "assets")
        model_package = torch.load(
            os.path.join(asset_dir, self.model_asset),
            map_location="cpu",
            weights_only=False,
        )
        power_package = torch.load(
            os.path.join(asset_dir, self.power_asset),
            map_location="cpu",
            weights_only=False,
        )
        self.eta_estimator = model_package["eta_estimator"]
        self.power_estimator = power_package["power_estimator"]
        self._freeze_estimators()
        self._register_package_buffers(model_package, "")
        self._register_package_buffers(power_package, "pow_")

    def train(self, mode=True):
        """Keep the surrogate in evaluation mode while allowing gradients through the
        trainable circuit parameters.
        """
        super().train(mode)
        self._set_estimators_eval()
        return self

    def _freeze_estimators(self):
        self._set_estimators_eval()
        for estimator in (self.eta_estimator, self.power_estimator):
            for parameter in estimator.parameters():
                parameter.requires_grad = False

    def _set_estimators_eval(self):
        self.eta_estimator.train(False)
        self.power_estimator.train(False)

    def _register_package_buffers(self, package, prefix):
        for key in ("X_max", "X_min", "Y_max", "Y_min"):
            self.register_buffer(
                f"{prefix}{key}", torch.as_tensor(package[key], dtype=torch.float32)
            )

    def _nominal_parameters(self):
        normalized = torch.zeros(
            (self.parameter_sets, self.X_max.numel()),
            dtype=self.rt_.dtype,
            device=self.device,
        )
        normalized[:, : self.rt_.shape[1]] = torch.sigmoid(self.rt_)
        return normalized * (self.X_max - self.X_min) + self.X_min

    def _sample_parameter_variation(self, physical_instances):
        """Sample device parameter variation independently for each physical circuit instance."""
        sample_count, epsilon = self.sampling
        nominal_parameters = self._nominal_parameters()
        if self.parameter_sets == 1:
            nominal_parameters = nominal_parameters.expand(physical_instances, -1)
        elif self.parameter_sets != physical_instances:
            raise ValueError(
                f"Circuit parameter sets ({self.parameter_sets}) do not match "
                f"physical instances ({physical_instances})."
            )
        variation = self.variation.sample(nominal_parameters, sample_count, epsilon)
        return nominal_parameters.unsqueeze(0) * variation

    def _eta_and_power(self, physical_instances):
        rt = self._sample_parameter_variation(physical_instances)
        normalized = self._normalise_extended(rt)
        eta_normalized = self.eta_estimator(normalized)
        eta = eta_normalized * (self.Y_max - self.Y_min) + self.Y_min
        if not power_requested(self.args):
            # Power is discarded unless a power objective asks for it; skipping the
            # estimator MLP keeps the forward output bit-identical.
            return eta, torch.zeros(eta.shape[0], eta.shape[1], dtype=eta.dtype, device=eta.device)
        power_normalized = self.power_estimator(normalized)
        power = power_normalized * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return eta, power.reshape(power.shape[0], power.shape[1], -1).mean(dim=-1)

    def reference_eta(self):
        """Return nominal transfer parameters for initializing the crossbar bias."""
        rt = self._nominal_parameters()
        normalized = self._normalise_extended(rt)
        eta_normalized = self.eta_estimator(normalized)
        return eta_normalized * (self.Y_max - self.Y_min) + self.Y_min

    def reference_operating_input(self):
        """Return a reference input voltage in the high-slope region of the transfer curve."""
        return self.reference_eta()[0, 2]

    def spice_subcircuit(self, name, parameter_set=0, **options):
        """Generate a SPICE subcircuit with the current nominal device parameters.

        Each primitive defines its own topology and ports. Primitives without a defined
        physical topology raise an error on export.
        """
        raise NotImplementedError(
            f"SPICE export is not implemented for {type(self).__name__}."
        )

    def spice_instance_ports(
        self,
        input_node,
        output_node,
        vdd_node="VDD",
        vss_node="VSS",
        gnd_node="GND",
    ):
        """Order the nodes of a SPICE instance according to the primitive's port roles."""
        nodes = {
            "input": input_node,
            "output": output_node,
            "vdd": vdd_node,
            "vss": vss_node,
            "gnd": gnd_node,
        }
        try:
            return [nodes[role] for role in self.spice_port_roles]
        except KeyError as error:
            raise ValueError(
                f"Unsupported SPICE port role {error.args[0]!r} for {type(self).__name__}."
            ) from error

    @classmethod
    def _validate_spice_identifier(cls, value, field):
        """Validate a SPICE netlist or model identifier."""
        if not isinstance(value, str) or cls._SPICE_IDENTIFIER.fullmatch(value) is None:
            raise ValueError(f"Invalid SPICE {field}: {value!r}.")
        return value

    def _spice_parameter_values(self, parameter_set):
        """Return the nominal physical device values for one set of learned parameters."""
        if isinstance(parameter_set, bool) or not isinstance(parameter_set, int):
            raise TypeError("parameter_set must be an integer.")
        if not 0 <= parameter_set < self.parameter_sets:
            raise IndexError(
                f"parameter_set {parameter_set} is outside [0, {self.parameter_sets})."
            )
        return self._nominal_parameters()[parameter_set].detach().cpu().tolist()

    @staticmethod
    def _spice_value(value, scale=1.0, suffix=""):
        """Format a SPICE value using consistent precision and engineering units."""
        return f"{value / scale:.9g}{suffix}"

    def forward(self, signal, active_device_mask=None):
        """Compute all signal outputs and sum power over the active devices.

        The mask identifies the devices that remain after hard thresholding and pruning. It
        applies to the power calculation only.
        """
        if signal.ndim != 3:
            raise ValueError(
                "Circuit signal must have shape [sample, batch, instance]."
            )
        sample_count, _, physical_instances = signal.shape
        expected_samples, _ = self.sampling
        if sample_count != expected_samples:
            raise ValueError(
                f"Circuit received {sample_count} samples, but stage {self.stage} requires {expected_samples}."
            )
        eta, per_instance_power = self._eta_and_power(physical_instances)
        output = self._transfer(signal, eta)
        if not torch.isfinite(output).all():
            raise FloatingPointError("Circuit output contains non-finite values.")
        if not power_requested(self.args):
            return ForwardResult(output=output, power=self._zero_power(output))
        if active_device_mask is None:
            active_device_mask = torch.ones(
                physical_instances, dtype=signal.dtype, device=self.device
            )
        else:
            active_device_mask = active_device_mask.to(
                device=self.device, dtype=signal.dtype
            )
            if active_device_mask.shape != (physical_instances,):
                raise ValueError(
                    "active_device_mask must have one value per physical instance."
                )
        power_per_sample = (per_instance_power * active_device_mask.unsqueeze(0)).sum(
            dim=1
        )
        power = power_per_sample.sum() / (
            power_per_sample.shape[0] + torch.finfo(power_per_sample.dtype).tiny
        )
        if not torch.isfinite(power):
            raise FloatingPointError("Circuit power is non-finite.")
        if power < 0:
            raise FloatingPointError("Circuit power must be non-negative.")
        return ForwardResult(output=output, power=power)

    @staticmethod
    def _zero_power(output):
        """Return zero power as a scalar on the output device with the output dtype when
        power is disabled.
        """
        return output.new_zeros(())

    def SetParameter(self, name, value):
        if name == "args":
            self.args = value
        elif name == "stage":
            if value not in {"train", "valid", "test"}:
                raise ValueError(f"Unsupported circuit stage: {value}.")
            self.stage = value

    @abstractmethod
    def _normalise_extended(self, rt):
        """Convert device parameters with manufacturing variation into surrogate model inputs."""

    @abstractmethod
    def _transfer(self, signal, eta):
        """Apply the circuit transfer function using the surrogate parameters."""


class InverterCircuit(NonlinearCircuit):
    """EA_LNC inverter for negative-weight paths."""

    initial_rt = (
        -7.0846,
        -2.5695,
        16.1542,
        13.8696,
        2.7635,
        1.2674,
        -10.4803,
        -6.4386,
        5.1005,
    )
    device_inventory = {"resistor": 3, "transistor": 3}
    spice_port_roles = ("input", "output", "vdd", "gnd")
    model_asset = "neg_param.package"
    power_asset = "neg_power.package"

    def spice_subcircuit(
        self,
        name="inverter",
        parameter_set=0,
        transistor_model="nmos",
    ):
        """Generate the three-stage resistively loaded NMOS inverter subcircuit.

        The ports are ``VP VN VDD GND`` in that order. The internal stage nodes are ``n1``
        and ``n2``. Resistance is in kΩ and transistor width and length are in µm.
        """
        name = self._validate_spice_identifier(name, "subcircuit name")
        transistor_model = self._validate_spice_identifier(
            transistor_model, "transistor model"
        )
        values = self._spice_parameter_values(parameter_set)
        r1, r2, r3, w1, l1, w2, l2, w3, l3 = values[:9]
        resistor = lambda value: self._spice_value(value, scale=1e3, suffix="k")
        length = lambda value: self._spice_value(value, scale=1e-6, suffix="u")

        lines = [
            f"* Three-stage resistive-load NMOS inverter, parameter set {parameter_set}",
            f".SUBCKT {name} VP VN VDD GND",
            f"R1 VDD n1 {resistor(r1)}",
            f"M1 n1 VP GND GND {transistor_model} W={length(w1)} L={length(l1)}",
            f"R2 VDD n2 {resistor(r2)}",
            f"M2 n2 n1 GND GND {transistor_model} W={length(w2)} L={length(l2)}",
            f"R3 VDD VN {resistor(r3)}",
            f"M3 VN n2 GND GND {transistor_model} W={length(w3)} L={length(l3)}",
            f".ENDS {name}",
        ]
        return "\n".join(lines) + "\n"

    def _normalise_extended(self, rt):
        extended = torch.stack(
            [
                rt[..., 0],
                rt[..., 1],
                rt[..., 2],
                rt[..., 3],
                rt[..., 4],
                rt[..., 5],
                rt[..., 6],
                rt[..., 7],
                rt[..., 8],
                rt[..., 3] / (rt[..., 4] + torch.finfo(rt.dtype).tiny),
                rt[..., 5] / (rt[..., 6] + torch.finfo(rt.dtype).tiny),
                rt[..., 7] / (rt[..., 8] + torch.finfo(rt.dtype).tiny),
            ],
            dim=-1,
        )
        return (extended - self.X_min) / (
            self.X_max - self.X_min + torch.finfo(extended.dtype).tiny
        )

    def _transfer(self, signal, eta):
        eta = eta.unsqueeze(1)
        return -(
            eta[..., 0] + eta[..., 1] * torch.tanh((signal - eta[..., 2]) * eta[..., 3])
        )


class TanhCircuit(NonlinearCircuit):
    """EA_LNC hyperbolic tangent activation circuit."""

    initial_rt = (12.7865, -3.1871, -10.4537, 10.5460, -5.8496, 4.2337)
    device_inventory = {"resistor": 2, "transistor": 2}
    spice_port_roles = ("input", "output", "vdd", "vss")
    model_asset = "act_model_package"
    power_asset = "act_power_model_package"

    def spice_subcircuit(
        self,
        name="tanh",
        parameter_set=0,
        transistor_model="nmos",
    ):
        """Generate the two-stage resistively loaded NMOS tanh subcircuit.

        The ports are ``VZ VA VDD VSS`` in that order. The drain of ``T1`` at ``n1`` drives
        the gate of ``T2``. The top-level netlist sets ``VSS``, the negative supply shown as
        -1 V in the schematic.
        """
        name = self._validate_spice_identifier(name, "subcircuit name")
        transistor_model = self._validate_spice_identifier(
            transistor_model, "transistor model"
        )
        values = self._spice_parameter_values(parameter_set)
        r1, r2, w1, l1, w2, l2 = values[:6]
        resistor = lambda value: self._spice_value(value, scale=1e3, suffix="k")
        length = lambda value: self._spice_value(value, scale=1e-6, suffix="u")

        lines = [
            f"* Two-stage resistive-load NMOS tanh circuit, parameter set {parameter_set}",
            f".SUBCKT {name} VZ VA VDD VSS",
            f"R1 VDD n1 {resistor(r1)}",
            f"M1 n1 VZ VSS VSS {transistor_model} W={length(w1)} L={length(l1)}",
            f"R2 VDD VA {resistor(r2)}",
            f"M2 VA n1 VSS VSS {transistor_model} W={length(w2)} L={length(l2)}",
            f".ENDS {name}",
        ]
        return "\n".join(lines) + "\n"

    def _normalise_extended(self, rt):
        extended = torch.stack(
            [
                rt[..., 0],
                rt[..., 1],
                rt[..., 2],
                rt[..., 3],
                rt[..., 4],
                rt[..., 5],
                rt[..., 1] / (rt[..., 0] + torch.finfo(rt.dtype).tiny),
                rt[..., 3] / (rt[..., 2] + torch.finfo(rt.dtype).tiny),
                rt[..., 5] / (rt[..., 4] + torch.finfo(rt.dtype).tiny),
            ],
            dim=-1,
        )
        return (extended - self.X_min) / (
            self.X_max - self.X_min + torch.finfo(extended.dtype).tiny
        )

    def _transfer(self, signal, eta):
        eta = eta.unsqueeze(1)
        return eta[..., 0] + eta[..., 1] * torch.tanh(
            (signal - eta[..., 2]) * eta[..., 3]
        )


class HardSigmoidCircuit(NonlinearCircuit):
    """EA_LNC hard sigmoid activation circuit."""

    initial_rt = (-0.3766, 0.5192, 4.6293)
    device_inventory = {"resistor": 1, "transistor": 1}
    spice_port_roles = ("input", "output", "vdd", "gnd")
    model_asset = "hard_sigmoid_param.package"
    power_asset = "hard_sigmoid_power.package"

    def spice_subcircuit(
        self,
        name="hard_sigmoid",
        parameter_set=0,
        transistor_model="nmos",
    ):
        """Generate the hard sigmoid subcircuit with a high-side NMOS and a pull-down
        resistor.

        The ports are ``VZ VA VDD GND`` in that order. The drain connects to ``VDD``, the
        source to ``VA``, and the gate to ``VZ``. The trainable pull-down resistor ``R1``
        connects ``VA`` to ``GND``. Parameters follow the surrogate input order ``R, W, L``.
        """
        name = self._validate_spice_identifier(name, "subcircuit name")
        transistor_model = self._validate_spice_identifier(
            transistor_model, "transistor model"
        )
        resistance, width, length_value = self._spice_parameter_values(parameter_set)[:3]
        resistor = self._spice_value(resistance, scale=1e3, suffix="k")
        length = lambda value: self._spice_value(value, scale=1e-6, suffix="u")

        lines = [
            f"* High-side NMOS hard sigmoid circuit, parameter set {parameter_set}",
            f".SUBCKT {name} VZ VA VDD GND",
            f"M1 VDD VZ VA GND {transistor_model} W={length(width)} L={length(length_value)}",
            f"R1 VA GND {resistor}",
            f".ENDS {name}",
        ]
        return "\n".join(lines) + "\n"

    def _normalise_extended(self, rt):
        extended = torch.stack(
            [
                rt[..., 0],
                rt[..., 1],
                rt[..., 2],
                rt[..., 1] / (rt[..., 2] + torch.finfo(rt.dtype).tiny),
            ],
            dim=-1,
        )
        return (extended - self.X_min) / (
            self.X_max - self.X_min + torch.finfo(extended.dtype).tiny
        )

    def reference_operating_input(self):
        """Use the midpoint of the hard sigmoid's constant-slope linear region."""
        eta = self.reference_eta()[0]
        return (eta[2] + eta[3]) / 2.0

    def _transfer(self, signal, eta):
        eta = eta.unsqueeze(1)
        linear = eta[..., 0] + (eta[..., 1] - eta[..., 0]) / (
            eta[..., 3] - eta[..., 2] + torch.finfo(eta.dtype).tiny
        ) * (signal - eta[..., 2])
        return torch.where(
            signal < eta[..., 2],
            eta[..., 0],
            torch.where(signal <= eta[..., 3], linear, eta[..., 1]),
        )


class PReLUCircuit(NonlinearCircuit):
    """EA_LNC soft pReLU activation circuit."""

    initial_rt = (-1.9557, -20.1581, 16.6486, -14.3472, 16.6550, -21.7591)
    device_inventory = {"resistor": 4, "transistor": 1}
    spice_port_roles = ("input", "output", "gnd")
    model_asset = "ReLU_param.package"
    power_asset = "ReLU_power.package"

    def spice_subcircuit(
        self,
        name="prelu",
        parameter_set=0,
        transistor_model="nmos",
    ):
        """Generate the soft pReLU subcircuit with a resistor bridge and one transistor.

        The ports are ``VZ VA GND`` in that order. The node ``VG`` between ``R1`` and ``R2``
        drives the transistor gate. The node between ``R3`` and ``R4`` is grounded. The
        EA_LNC parameters ``RH, RL, RD, RB, W, L`` correspond to ``R1, R2, R3, R4, TRe.W,
        TRe.L`` in the schematic, respectively.
        """
        name = self._validate_spice_identifier(name, "subcircuit name")
        transistor_model = self._validate_spice_identifier(
            transistor_model, "transistor model"
        )
        values = self._spice_parameter_values(parameter_set)
        r_high, r_low, r_drain, r_base, width, length_value = values[:6]
        resistor = lambda value: self._spice_value(value, scale=1e3, suffix="k")
        length = lambda value: self._spice_value(value, scale=1e-6, suffix="u")

        lines = [
            f"* Bridge-resistor NMOS soft pReLU circuit, parameter set {parameter_set}",
            f".SUBCKT {name} VZ VA GND",
            f"R1 VZ VG {resistor(r_high)}",
            f"R2 VG VA {resistor(r_low)}",
            f"R3 VZ GND {resistor(r_drain)}",
            f"R4 GND VA {resistor(r_base)}",
            f"M1 VA VG VZ GND {transistor_model} W={length(width)} L={length(length_value)}",
            f".ENDS {name}",
        ]
        return "\n".join(lines) + "\n"

    def _normalise_extended(self, rt):
        extended = torch.stack(
            [
                rt[..., 0],
                rt[..., 1],
                rt[..., 2],
                rt[..., 3],
                rt[..., 4],
                rt[..., 5],
                rt[..., 0] / (rt[..., 1] + torch.finfo(rt.dtype).tiny),
                rt[..., 4] / (rt[..., 5] + torch.finfo(rt.dtype).tiny),
            ],
            dim=-1,
        )
        return (extended - self.X_min) / (
            self.X_max - self.X_min + torch.finfo(extended.dtype).tiny
        )

    def _transfer(self, signal, eta):
        eta = eta.unsqueeze(1)
        beta = eta[..., 4]
        softplus = torch.log(1.0 + torch.exp(beta * (signal - eta[..., 2]))) / (
            beta + torch.finfo(eta.dtype).tiny
        )
        return (
            eta[..., 0] * (signal - eta[..., 2]) + eta[..., 1] * softplus + eta[..., 3]
        )


class SigmoidCircuit(NonlinearCircuit):
    """EA_LNC sigmoid activation circuit."""

    initial_rt = (-11.7481, -0.3905, -9.1690, 0.3913, 7.2004, 15.5558)
    device_inventory = {"resistor": 3, "transistor": 2}
    spice_port_roles = ("input", "output", "vdd", "gnd")
    interstage_resistance = 100.0
    model_asset = "sigmoid_param.package"
    power_asset = "sigmoid_power.package"

    def spice_subcircuit(
        self,
        name="sigmoid",
        parameter_set=0,
        transistor_model="nmos",
    ):
        """Generate the sigmoid subcircuit with its fixed 100 Ω interstage resistor.

        The ports are ``VZ VA VDD GND`` in that order. ``R100`` connects the drain of ``T1``
        at ``n1`` to the gate of ``T2`` at ``vg2``. Its value stays fixed during training
        and across ``parameter_set`` selections. The top-level netlist supplies the
        schematic's ``VSS=0`` reference.
        """
        name = self._validate_spice_identifier(name, "subcircuit name")
        transistor_model = self._validate_spice_identifier(
            transistor_model, "transistor model"
        )
        values = self._spice_parameter_values(parameter_set)
        r1, r2, w1, l1, w2, l2 = values[:6]
        resistor = lambda value: self._spice_value(value, scale=1e3, suffix="k")
        length = lambda value: self._spice_value(value, scale=1e-6, suffix="u")
        interstage_resistance = self._spice_value(self.interstage_resistance)

        lines = [
            f"* Two-stage resistive-load NMOS sigmoid circuit, parameter set {parameter_set}",
            f".SUBCKT {name} VZ VA VDD GND",
            f"R1 VDD n1 {resistor(r1)}",
            f"M1 n1 VZ GND GND {transistor_model} W={length(w1)} L={length(l1)}",
            f"R100 n1 vg2 {interstage_resistance}",
            f"R2 VDD VA {resistor(r2)}",
            f"M2 VA vg2 GND GND {transistor_model} W={length(w2)} L={length(l2)}",
            f".ENDS {name}",
        ]
        return "\n".join(lines) + "\n"

    def _normalise_extended(self, rt):
        extended = torch.stack(
            [
                rt[..., 0],
                rt[..., 1],
                rt[..., 2],
                rt[..., 3],
                rt[..., 4],
                rt[..., 5],
                rt[..., 2] / (rt[..., 3] + torch.finfo(rt.dtype).tiny),
                rt[..., 4] / (rt[..., 5] + torch.finfo(rt.dtype).tiny),
            ],
            dim=-1,
        )
        return (extended - self.X_min) / (
            self.X_max - self.X_min + torch.finfo(extended.dtype).tiny
        )

    def _transfer(self, signal, eta):
        eta = eta.unsqueeze(1)
        return eta[..., 0] + eta[..., 1] * torch.sigmoid(
            (signal - eta[..., 2]) * eta[..., 3]
        )


ACTIVATION_CIRCUITS = {
    "tanh": TanhCircuit,
    "hard_sigmoid": HardSigmoidCircuit,
    "prelu": PReLUCircuit,
    "sigmoid": SigmoidCircuit,
}


def create_activation_circuit(name, args, parameter_sets=1, variation=None):
    """Create the activation circuit named in the configuration."""
    try:
        return ACTIVATION_CIRCUITS[name](
            args, parameter_sets=parameter_sets, variation=variation
        )
    except KeyError as error:
        supported = ", ".join(sorted(ACTIVATION_CIRCUITS))
        raise ValueError(
            f"Unsupported activation circuit: {name}. Supported: {supported}."
        ) from error
