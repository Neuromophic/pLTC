"""pNN architectures and device counts."""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .layer import GatedStateCell, pLayer
from .primitives.circuits import (
    ForwardResult,
    InverterCircuit,
    NonlinearCircuit,
    create_activation_circuit,
)
from .primitives.variation import SequenceVariation, create_variation
from .primitives.crossbar import Crossbar
from .primitives.gated_rc import GatedRCBranch


class pNN(nn.Module):
    """Printed neural network built from crossbars and nonlinear circuits."""

    def __init__(self, topology, args):
        super().__init__()
        if len(topology) < 2:
            raise ValueError("Topology must contain at least input and output widths.")
        self.args = args
        self.stage = "train"
        self.topology = tuple(topology)
        self.layers = nn.ModuleList()
        self.crossbar_variation = create_variation(
            getattr(args, "crossbar_variation", "uniform")
        )
        self.circuit_variation = create_variation(
            getattr(args, "circuit_variation", "uniform")
        )
        self.activations, self.inverters = self._create_circuit_banks()
        self.bias_inverter = InverterCircuit(args, variation=self.circuit_variation)
        self._activation_slots = []
        for index, (n_in, n_out) in enumerate(zip(topology[:-1], topology[1:])):
            activation_slot = self._activation_slot(index)
            activation = (
                None if activation_slot is None else self.activations[activation_slot]
            )
            operating_input = (
                None
                if activation is None
                else float(activation.reference_operating_input().detach())
            )
            self._activation_slots.append(activation_slot)
            self.layers.append(
                pLayer(
                    n_in,
                    n_out,
                    args,
                    has_activation=activation is not None,
                    activation_operating_input=operating_input,
                    variation=self.crossbar_variation,
                )
            )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load older checkpoints that also stored baselines for pruning by ratio."""
        state_dict = state_dict.copy()
        state_dict.pop("_ratio_pruning_network_baseline", None)
        state_dict.pop("_ratio_pruning_layer_baselines", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def DrawArchitecture(self, path, show_pruned=False):
        """Draw the MLP topology and current pruning state as a PNG."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle

        path = Path(path)
        if path.suffix.lower() != ".png":
            raise ValueError("Architecture output must be a PNG file.")
        path.parent.mkdir(parents=True, exist_ok=True)

        weights = [
            layer.crossbar.theta.detach().cpu()[: layer.crossbar.n_in]
            for layer in self.layers
        ]
        layer_spacing = 2.4
        figure = Figure(figsize=(max(8.0, 2.75 * len(self.topology)), 5.5))
        FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        positions = {}
        for layer_index, width in enumerate(self.topology):
            vertical_positions = (
                [0.0]
                if width == 1
                else torch.linspace(1.0, -1.0, width).tolist()
            )
            for node_index, vertical in enumerate(vertical_positions):
                positions[(layer_index, node_index)] = (
                    layer_index * layer_spacing,
                    vertical,
                )

        active_nodes = [(weights[0] != 0).any(dim=1)]
        active_nodes.extend((weight != 0).any(dim=0) for weight in weights)
        for layer_index, (layer, weight) in enumerate(zip(self.layers, weights)):
            effective = weight != 0
            scale = weight.abs()[effective].max().item() if effective.any() else 1.0
            for source, target in torch.nonzero(effective, as_tuple=False).tolist():
                value = weight[source, target].item()
                axis.plot(
                    [
                        positions[(layer_index, source)][0],
                        positions[(layer_index + 1, target)][0],
                    ],
                    [
                        positions[(layer_index, source)][1],
                        positions[(layer_index + 1, target)][1],
                    ],
                    color="#2979ff" if value > 0 else "#ef6c00",
                    alpha=0.48,
                    linewidth=0.65 + 1.75 * abs(value) / scale,
                    solid_capstyle="round",
                    zorder=1,
                )

            if show_pruned:
                pruned = (
                    layer.crossbar.pruning_mask.detach().cpu()[: layer.crossbar.n_in]
                    == 0
                )
                for source, target in torch.nonzero(pruned, as_tuple=False).tolist():
                    axis.plot(
                        [
                            positions[(layer_index, source)][0],
                            positions[(layer_index + 1, target)][0],
                        ],
                        [
                            positions[(layer_index, source)][1],
                            positions[(layer_index + 1, target)][1],
                        ],
                        color="#a8b3c1",
                        linestyle=(0, (5, 4)),
                        linewidth=1.0,
                        zorder=0,
                    )

        for layer_index, width in enumerate(self.topology):
            for node_index in range(width):
                active = bool(active_nodes[layer_index][node_index].item())
                if not active and not show_pruned:
                    continue
                position = positions[(layer_index, node_index)]
                axis.add_patch(
                    Circle(
                        position,
                        radius=0.07,
                        facecolor="#163a5f" if active else "none",
                        edgecolor="#163a5f" if active else "#a8b3c1",
                        linestyle="solid" if active else "--",
                        linewidth=1.4,
                        zorder=2,
                    )
                )

        layer_names = [
            "Input",
            *[
                f"Hidden {index}"
                for index in range(1, len(self.topology) - 1)
            ],
            "Output",
        ]
        for layer_index, name in enumerate(layer_names):
            axis.text(
                layer_index * layer_spacing,
                1.18,
                name,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
                color="#0f172a",
            )

        legend_handles = [
            Line2D([], [], color="#2979ff", label="Positive weight"),
            Line2D([], [], color="#ef6c00", label="Negative weight"),
        ]
        if show_pruned:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="#a8b3c1",
                    markerfacecolor="none",
                    linestyle=(0, (5, 4)),
                    label="Pruned / inactive",
                )
            )
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(legend_handles),
        )
        figure.subplots_adjust(top=0.86, bottom=0.18, left=0.05, right=0.95)
        axis.set_xlim(-0.65, (len(self.topology) - 1) * layer_spacing + 0.65)
        axis.set_ylim(-1.18, 1.35)
        axis.set_aspect("auto")
        axis.axis("off")
        figure.savefig(path, dpi=220, bbox_inches="tight")
        return figure

    def _create_circuit_banks(self):
        layer_count = len(self.topology) - 1
        hidden_count = layer_count - 1
        sharing = self.args.circuit_sharing
        if sharing == "network":
            activations = (
                nn.ModuleList(
                    [
                        create_activation_circuit(
                            self.args.activation_circuit,
                            self.args,
                            variation=self.circuit_variation,
                        )
                    ]
                )
                if hidden_count
                else nn.ModuleList()
            )
            inverters = nn.ModuleList(
                [InverterCircuit(self.args, variation=self.circuit_variation)]
            )
        elif sharing == "layer":
            activations = nn.ModuleList(
                [
                    create_activation_circuit(
                        self.args.activation_circuit,
                        self.args,
                        variation=self.circuit_variation,
                    )
                    for _ in range(hidden_count)
                ]
            )
            inverters = nn.ModuleList(
                [
                    InverterCircuit(self.args, variation=self.circuit_variation)
                    for _ in range(layer_count)
                ]
            )
        elif sharing == "neuron":
            activations = nn.ModuleList(
                [
                    create_activation_circuit(
                        self.args.activation_circuit,
                        self.args,
                        parameter_sets=self.topology[index + 1],
                        variation=self.circuit_variation,
                    )
                    for index in range(hidden_count)
                ]
            )
            inverters = nn.ModuleList(
                [
                    InverterCircuit(self.args, variation=self.circuit_variation)
                    for _ in range(layer_count)
                ]
            )
        else:
            raise ValueError(f"Unsupported circuit sharing: {sharing}.")
        return activations, inverters

    def _activation_slot(self, layer_index):
        """Return the activation slot for a layer, or ``None`` for the output layer.

        With layer-level sharing, topology ``[2, 3, 4, 2]`` has two activations,
        ``[hidden_0, hidden_1]``, and three layer slots, ``[0, 1, None]``.
        """
        if layer_index == len(self.topology) - 2:
            return None
        return 0 if self.args.circuit_sharing == "network" else layer_index

    def _inverter(self, layer_index):
        """Return the layer's inverter parameter group. Each forward pass samples the
        physical instances independently.
        """
        return (
            self.inverters[0]
            if self.args.circuit_sharing == "network"
            else self.inverters[layer_index]
        )

    def _activation(self, layer_index):
        """Look up the layer's activation by slot so shared modules are registered only once."""
        slot = self._activation_slots[layer_index]
        return None if slot is None else self.activations[slot]

    def _bias_inverter_mask(self, device, dtype):
        """Return the hard mask for the shared bias inverter with a straight-through gradient."""
        relaxed_masks = []
        hard_masks = []
        for layer in self.layers:
            bias_theta = layer.crossbar.theta[-2]
            negative = (bias_theta < 0).to(dtype=bias_theta.dtype)
            hard_masks.append(negative.any().to(dtype=bias_theta.dtype))
            relaxed_masks.append(((1.0 - torch.sigmoid(bias_theta)) * negative).max())
        hard = torch.stack(hard_masks).amax()
        relaxed = torch.stack(relaxed_masks).amax()
        mask = hard.detach() + relaxed - relaxed.detach()
        return mask.reshape(1).to(device=device, dtype=dtype)

    def forward(self, signal):
        if signal.ndim != 2 or signal.shape[-1] != self.topology[0]:
            raise ValueError(
                f"Network input must have shape [batch, {self.topology[0]}]."
            )
        if self.stage in {"train", "valid"}:
            sample_count = self.args.N_train
        elif self.stage == "test":
            sample_count = self.args.N_test
        else:
            raise ValueError(f"Unsupported network stage: {self.stage}.")
        output = signal.unsqueeze(0).expand(sample_count, -1, -1)
        bias_signal = torch.ones_like(output[..., :1])
        bias_result = self.bias_inverter(
            bias_signal,
            active_device_mask=self._bias_inverter_mask(output.device, output.dtype),
        )
        total_power = bias_result.power
        for index, layer in enumerate(self.layers):
            result = layer(
                output,
                self._inverter(index),
                bias_result.output,
                self._activation(index),
            )
            output = result.output
            total_power = total_power + result.power
        if not torch.isfinite(output).all():
            raise FloatingPointError("Network output contains non-finite values.")
        if not torch.isfinite(total_power):
            raise FloatingPointError("Network power is non-finite.")
        if total_power < 0:
            raise FloatingPointError("Network power must be non-negative.")
        return ForwardResult(output=output, power=total_power)

    def spice_subcircuit(self, name="pnn", transistor_model="nmos"):
        """Export a pNN SPICE subcircuit for the current topology, parameters, and pruning
        masks.

        The subcircuit exposes input, output, positive bias, and supply ports. The caller
        supplies voltage sources, model cards, and simulation commands. Each crossbar
        retains its own resistances. Negative weights use the input inverters required by
        their layer, and the network shares a single inverter for negative bias.

        When pruning removes all signal inputs to a hidden neuron, the exporter omits its
        activation circuit and grounds the corresponding input to the next layer.
        """
        name = NonlinearCircuit._validate_spice_identifier(name, "subcircuit name")
        definitions = []
        defined_circuits = set()

        def add_circuit(circuit, circuit_name, parameter_set=0):
            """Emit each named nominal subcircuit definition once."""
            key = (id(circuit), parameter_set)
            if key not in defined_circuits:
                definitions.append(
                    circuit.spice_subcircuit(
                        name=circuit_name,
                        parameter_set=parameter_set,
                        transistor_model=transistor_model,
                    )
                )
                defined_circuits.add(key)
            return circuit_name

        inverter_names = {}
        for index, inverter in enumerate(self.inverters):
            inverter_names[id(inverter)] = add_circuit(
                inverter, f"inverter_{index}"
            )

        activation_names = {}
        for slot, activation in enumerate(self.activations):
            for parameter_set in range(activation.parameter_sets):
                suffix = "" if activation.parameter_sets == 1 else f"_{parameter_set}"
                activation_names[(slot, parameter_set)] = add_circuit(
                    activation,
                    f"activation_{slot}{suffix}",
                    parameter_set=parameter_set,
                )

        bias_required = any(
            layer.crossbar.has_negative_bias_connection() for layer in self.layers
        )
        if bias_required:
            add_circuit(self.bias_inverter, "bias_inverter")

        crossbar_names = []
        for index, layer in enumerate(self.layers):
            crossbar_name = f"crossbar_{index}"
            inverter_name = inverter_names[id(self._inverter(index))]
            definitions.append(
                layer.crossbar.spice_subcircuit(
                    name=crossbar_name,
                    inverter_subcircuit=inverter_name,
                )
            )
            crossbar_names.append(crossbar_name)

        input_nodes = [f"VIN_{index}" for index in range(self.topology[0])]
        output_nodes = [f"VOUT_{index}" for index in range(self.topology[-1])]
        ports = input_nodes + output_nodes + ["VBIAS_POS", "VDD", "VSS", "GND"]
        lines = [
            "* Printable pNN assembled from current nominal primitive parameters",
            f".SUBCKT {name} {' '.join(ports)}",
        ]
        bias_negative_node = "VBIAS_NEG" if bias_required else "GND"
        if bias_required:
            lines.append("XBIAS VBIAS_POS VBIAS_NEG VDD GND bias_inverter")

        current_inputs = input_nodes
        for index, layer in enumerate(self.layers):
            crossbar = layer.crossbar
            is_output_layer = index == len(self.layers) - 1
            raw_outputs = (
                output_nodes
                if is_output_layer
                else [f"L{index}_RAW_{output}" for output in range(crossbar.n_out)]
            )
            instance_ports = (
                current_inputs
                + ["VBIAS_POS", bias_negative_node]
                + raw_outputs
                + ["VDD", "GND", crossbar_names[index]]
            )
            lines.append(f"XCB_{index} {' '.join(instance_ports)}")

            if is_output_layer:
                continue

            activation = self._activation(index)
            active_mask = crossbar.active_activation_mask().detach().cpu().bool()
            next_inputs = []
            for output, raw_output in enumerate(raw_outputs):
                if not active_mask[output]:
                    next_inputs.append("GND")
                    continue
                activation_output = f"L{index}_OUT_{output}"
                parameter_set = output if activation.parameter_sets > 1 else 0
                activation_name = activation_names[
                    (self._activation_slots[index], parameter_set)
                ]
                activation_ports = activation.spice_instance_ports(
                    raw_output,
                    activation_output,
                )
                lines.append(
                    f"XACT_{index}_{output} {' '.join(activation_ports)} {activation_name}"
                )
                next_inputs.append(activation_output)
            current_inputs = next_inputs

        lines.append(f".ENDS {name}")
        return "\n".join(definitions + ["\n".join(lines) + "\n"])

    def GetParam(self, group_str="w,n"):
        groups = {
            group.strip().lower() for group in group_str.split(",") if group.strip()
        }
        parameters = []
        if {"w", "weight"} & groups:
            parameters.extend(layer.crossbar.theta_ for layer in self.layers)
        if {"n", "nonlinear"} & groups:
            parameters.extend(circuit.rt_ for circuit in self.activations)
            parameters.extend(circuit.rt_ for circuit in self.inverters)
            parameters.append(self.bias_inverter.rt_)
        return parameters

    def checkpoint_runtime_state(self):
        """Return the physical sampling state stored separately from ``state_dict``."""
        return {"stage": self.stage}

    def load_checkpoint_runtime_state(self, state):
        """Restore the physical sampling state from a checkpoint."""
        if set(state) != {"stage"}:
            raise ValueError("Invalid pNN runtime state.")
        self.SetParameter("stage", state["stage"])

    def DeviceCount(self):
        """Count the devices required by the connections that survive hard thresholding."""
        counts = {}
        for index in range(len(self.layers)):
            for name, quantity in self._layer_device_count(index).items():
                if name != "total":
                    counts[name] = counts.get(name, 0) + quantity

        bias_inverter_instances = int(
            any(layer.crossbar.has_negative_bias_connection() for layer in self.layers)
        )
        counts["inverter_instances"] += bias_inverter_instances
        for device_type, quantity in self.bias_inverter.device_inventory.items():
            counts[device_type] = (
                counts.get(device_type, 0) + quantity * bias_inverter_instances
            )
        counts["total"] = sum(
            quantity
            for name, quantity in counts.items()
            if name
            not in {"crossbar_instances", "activation_instances", "inverter_instances"}
        )
        return counts

    def _unpruned_device_counts(self):
        """Count the unpruned devices in the network and in each layer with pruning masks
        temporarily disabled.
        """
        masks = [layer.crossbar.pruning_mask.clone() for layer in self.layers]
        try:
            with torch.no_grad():
                for layer in self.layers:
                    layer.crossbar.pruning_mask.fill_(1.0)
            return self.DeviceCount(), [
                self._layer_device_count(index) for index in range(len(self.layers))
            ]
        finally:
            with torch.no_grad():
                for layer, mask in zip(self.layers, masks):
                    layer.crossbar.pruning_mask.copy_(mask)

    def _layer_device_count(self, layer_index):
        """Count the crossbar, input inverters, and activation circuits belonging to one layer."""
        counts = {
            "crossbar_instances": 0,
            "activation_instances": 0,
            "inverter_instances": 0,
            "resistor": 0,
            "transistor": 0,
        }

        def add_circuit_inventory(circuit, instances):
            for device_type, quantity in circuit.device_inventory.items():
                counts[device_type] = counts.get(device_type, 0) + quantity * instances

        layer = self.layers[layer_index]
        layer_counts = layer.crossbar.device_count()
        crossbar_instances = layer_counts["crossbar_instances"]
        inverter_instances = layer_counts["inverter_instances"]
        counts["crossbar_instances"] = crossbar_instances
        counts["inverter_instances"] = inverter_instances
        counts["resistor"] = (
            crossbar_instances * layer.crossbar.device_inventory["resistor"]
        )
        add_circuit_inventory(self._inverter(layer_index), inverter_instances)

        activation = self._activation(layer_index)
        if activation is not None:
            activation_instances = int(
                layer.crossbar.active_activation_mask().sum().item()
            )
            counts["activation_instances"] = activation_instances
            add_circuit_inventory(activation, activation_instances)

        counts["total"] = sum(
            quantity
            for name, quantity in counts.items()
            if name
            not in {"crossbar_instances", "activation_instances", "inverter_instances"}
        )
        return counts

    def _prune_layer_columns(self, layer_index, columns):
        """Prune entire output columns from the specified layer."""
        crossbar = self.layers[layer_index].crossbar
        columns = torch.as_tensor(
            columns, dtype=torch.bool, device=crossbar.pruning_mask.device
        )
        if columns.shape != (crossbar.n_out,):
            raise ValueError(f"Pruning columns must have shape ({crossbar.n_out},).")
        positions = torch.zeros_like(crossbar.pruning_mask, dtype=torch.bool)
        positions[:, columns] = True
        return crossbar.prune_positions(positions)

    def _prune_layer_input_rows(self, layer_index, rows):
        """Prune signal input rows in the specified layer, preserving its bias and reference
        rows.
        """
        crossbar = self.layers[layer_index].crossbar
        rows = torch.as_tensor(
            rows, dtype=torch.bool, device=crossbar.pruning_mask.device
        )
        if rows.shape != (crossbar.n_in,):
            raise ValueError(f"Pruning rows must have shape ({crossbar.n_in},).")
        positions = torch.zeros_like(crossbar.pruning_mask, dtype=torch.bool)
        positions[: crossbar.n_in][rows] = True
        return crossbar.prune_positions(positions)

    def _propagate_pruning(self):
        """Remove neurons with no input and connections with no downstream consumers until
        the topology stops changing.
        """
        propagated = 0
        while True:
            changed = 0
            for index, layer in enumerate(self.layers):
                crossbar = layer.crossbar
                empty_columns = (crossbar.theta.detach()[: crossbar.n_in] == 0).all(
                    dim=0
                )
                changed += self._prune_layer_columns(index, empty_columns)

            for index in range(1, len(self.layers)):
                crossbar = self.layers[index].crossbar
                empty_rows = (crossbar.theta.detach()[: crossbar.n_in] == 0).all(dim=1)
                changed += self._prune_layer_columns(index - 1, empty_rows)

            for index in range(len(self.layers) - 1):
                crossbar = self.layers[index].crossbar
                empty_columns = (crossbar.theta.detach() == 0).all(dim=0)
                changed += self._prune_layer_input_rows(index + 1, empty_columns)

            propagated += changed
            if changed == 0:
                return propagated

    @staticmethod
    def _reduction_ratio(before, after):
        denominator = before["total"]
        if denominator == 0:
            return 0.0
        return (denominator - after["total"]) / denominator

    def _pruning_report(
        self,
        mode,
        value,
        scope,
        before,
        direct_pruned,
        propagated_pruned,
        stop_reason,
        baseline=None,
        requested_ratio=None,
        target_reached=True,
        per_layer=None,
    ):
        after = self.DeviceCount()
        reduction_baseline = before if baseline is None else baseline
        report = {
            "mode": mode,
            "scope": scope,
            "value": value,
            "before": before,
            "baseline": baseline,
            "after": after,
            "requested_ratio": requested_ratio,
            "achieved_ratio": self._reduction_ratio(reduction_baseline, after),
            "target_reached": target_reached,
            "stop_reason": stop_reason,
            "direct_pruned_connections": direct_pruned,
            "propagated_pruned_connections": propagated_pruned,
        }
        if per_layer is not None:
            report["per_layer"] = per_layer
        return report

    def _active_candidates(self, layer_index=None):
        """Return prunable connections ordered by absolute conductance, layer, row, and column."""
        layer_indices = (
            range(len(self.layers)) if layer_index is None else [layer_index]
        )
        candidates = []
        for index in layer_indices:
            crossbar = self.layers[index].crossbar
            theta = crossbar.theta.detach()
            active = crossbar.pruning_mask.bool() & (theta != 0)
            for row, column in torch.nonzero(active, as_tuple=False).tolist():
                candidates.append((abs(theta[row, column].item()), index, row, column))
        return sorted(candidates)

    def _prune_candidate(self, candidate):
        """Prune a single connection from the ordered candidates."""
        _, layer_index, row, column = candidate
        crossbar = self.layers[layer_index].crossbar
        positions = torch.zeros_like(crossbar.pruning_mask, dtype=torch.bool)
        positions[row, column] = True
        return crossbar.prune_positions(positions)

    def _network_ratio_pruning(self, value):
        before = self.DeviceCount()
        baseline, _ = self._unpruned_device_counts()
        direct_pruned = propagated_pruned = 0
        stop_reason = "target_reached"
        while self._reduction_ratio(baseline, self.DeviceCount()) < value:
            candidates = self._active_candidates()
            if not candidates:
                stop_reason = "no_prunable_connection"
                break
            direct_pruned += self._prune_candidate(candidates[0])
            propagated_pruned += self._propagate_pruning()
        after = self.DeviceCount()
        target_reached = self._reduction_ratio(baseline, after) >= value
        return self._pruning_report(
            mode="ratio",
            value=value,
            scope="network",
            before=before,
            baseline=baseline,
            direct_pruned=direct_pruned,
            propagated_pruned=propagated_pruned,
            stop_reason=stop_reason,
            requested_ratio=value,
            target_reached=target_reached,
        )

    def _layer_ratio_report(self, before, value):
        report = {}
        for index, before_count in enumerate(before):
            after_count = self._layer_device_count(index)
            achieved_ratio = self._reduction_ratio(before_count, after_count)
            report[str(index)] = {
                "before": before_count,
                "after": after_count,
                "achieved_ratio": achieved_ratio,
                "target_reached": achieved_ratio >= value,
            }
        return report

    def _layer_ratio_pruning(self, value):
        global_before = self.DeviceCount()
        global_baseline, layer_baselines = self._unpruned_device_counts()
        direct_pruned = propagated_pruned = 0
        stop_reason = "target_reached"
        while True:
            per_layer = self._layer_ratio_report(layer_baselines, value)
            unmet = [
                index
                for index in range(len(self.layers))
                if not per_layer[str(index)]["target_reached"]
            ]
            if not unmet:
                break

            choices = []
            for index in unmet:
                candidates = self._active_candidates(index)
                if candidates:
                    choices.append(
                        (per_layer[str(index)]["achieved_ratio"], index, candidates[0])
                    )
            if not choices:
                stop_reason = "no_prunable_connection"
                break

            _, _, candidate = min(choices, key=lambda item: (item[0], item[1], item[2]))
            direct_pruned += self._prune_candidate(candidate)
            propagated_pruned += self._propagate_pruning()

        per_layer = self._layer_ratio_report(layer_baselines, value)
        target_reached = all(values["target_reached"] for values in per_layer.values())
        return self._pruning_report(
            mode="ratio",
            value=value,
            scope="layer",
            before=global_before,
            baseline=global_baseline,
            direct_pruned=direct_pruned,
            propagated_pruned=propagated_pruned,
            stop_reason=stop_reason,
            requested_ratio=value,
            target_reached=target_reached,
            per_layer=per_layer,
        )

    def Prune(self, mode, value, scope=None):
        """Permanently prune connections by threshold or ratio and return a serializable report."""
        if mode not in {"threshold", "ratio"}:
            raise ValueError(f"Unsupported pruning mode: {mode}.")
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Pruning value must be a finite number.") from error
        if not math.isfinite(value):
            raise ValueError("Pruning value must be a finite number.")

        if mode == "ratio":
            if scope not in {"network", "layer"}:
                raise ValueError("Ratio pruning requires scope 'network' or 'layer'.")
            if not 0.0 <= value <= 1.0:
                raise ValueError("Ratio pruning value must be between 0 and 1.")
            if scope == "network":
                return self._network_ratio_pruning(value)
            return self._layer_ratio_pruning(value)

        if scope is not None:
            raise ValueError("Threshold pruning does not accept a scope.")
        if value < 0.0:
            raise ValueError("Threshold pruning value must be non-negative.")

        before = self.DeviceCount()
        direct_pruned = 0
        for layer in self.layers:
            crossbar = layer.crossbar
            theta = crossbar.theta.detach()
            positions = (
                crossbar.pruning_mask.bool() & (theta != 0) & (theta.abs() <= value)
            )
            direct_pruned += crossbar.prune_positions(positions)
        propagated_pruned = self._propagate_pruning()
        return self._pruning_report(
            mode="threshold",
            value=value,
            scope=None,
            before=before,
            direct_pruned=direct_pruned,
            propagated_pruned=propagated_pruned,
            stop_reason="threshold_applied",
        )

    def SetParameter(self, name, value):
        if name == "args":
            self.args = value
        elif name == "stage":
            if value not in {"train", "valid", "test"}:
                raise ValueError(f"Unsupported network stage: {value}.")
            self.stage = value
        for layer in self.layers:
            layer.SetParameter(name, value)
        for circuit in (
            list(self.activations) + list(self.inverters) + [self.bias_inverter]
        ):
            circuit.SetParameter(name, value)


@dataclass
class TemporalNetworkResult:
    """Final capacitor state and classification voltages. Total power is outside the scope
    of this result.
    """
    state: torch.Tensor
    output: torch.Tensor
    trace: dict = None
    output_sequence: torch.Tensor = None


GatedNetworkResult = TemporalNetworkResult


def initial_state_range(args):
    """Return the uniform initial-state range [0, HIGH]. The reference implementation
    defaults to [0, 1].
    """
    upper = getattr(args, "initial_range", None)
    if upper is None:
        return 0.0, 1.0
    if not math.isfinite(upper) or upper <= 0:
        raise ValueError("Require a finite positive --initial-range HIGH.")
    return 0.0, float(upper)


def _freeze_unselected_temporal_circuits(model):
    groups = {part.strip().lower() for part in getattr(model.args, "learnable_group", "w,n").split(",")}
    if not groups & {"n", "nonlinear"}:
        for module in model.modules():
            if isinstance(module, NonlinearCircuit):
                module.rt_.requires_grad_(False)


def _temporal_device_inventory(layer_specs, bias_inverter, components):
    """Count devices in one temporal network. Count physical circuits separately even when
    they share parameters or an activation produces a constant output.
    """
    def circuit_counts(circuit, instances):
        return {kind: quantity * instances for kind, quantity in circuit.device_inventory.items()}

    def add(counts, additions):
        for key, value in additions.items():
            counts[key] = counts.get(key, 0) + value

    for name, layer, inverter, activation in layer_specs:
        counts = dict(layer.crossbar.device_count())
        counts["resistor"] = counts["crossbar_instances"] * layer.crossbar.device_inventory["resistor"]
        add(counts, circuit_counts(inverter, counts["inverter_instances"]))
        if activation is not None:
            counts["activation_instances"] = layer.crossbar.n_out
            add(counts, circuit_counts(activation, layer.crossbar.n_out))
        components[name] = counts
    bias_instances = int(any(layer.crossbar.has_negative_bias_connection()
                             for _, layer, _, _ in layer_specs))
    components["bias_inverter"] = dict(circuit_counts(bias_inverter, bias_instances),
                                       inverter_instances=bias_instances)
    totals = dict(resistor=0, transistor=0, capacitor=0,
                  crossbar_instances=0, activation_instances=0, inverter_instances=0)
    for counts in components.values():
        counts["total"] = sum(counts.get(kind, 0) for kind in ("resistor", "transistor", "capacitor"))
        add(totals, counts)
    return dict(version=1, counts=totals, components=components, scope="single modeled temporal network",
                counting_policy="Hard-threshold crossbar connections; input inverters per crossbar input; "
                "one shared negative-bias inverter if used. All enabled nonlinear outputs and state branches "
                "retained; no graph-level dead-state pruning or constant-source folding. Parameter sharing "
                "does not merge physical circuits. Counts exclude time steps, batches and Monte Carlo chips.",
                unmodeled_components=["supply/reference voltage generation", "input encoding and output sensing",
                                      "additional finite-load drivers/buffers", "interconnect parasitics"],
                full_system_inventory_complete=False)


class pGatedNN(nn.Module):
    """Model a gated network with topology [input, state, output] and four explicit
    activation circuits.

    ``cell.readout`` reads the capacitor state. The target and gate activations drive the
    two branch inputs. As in the public pRNN implementation, ``output_activation`` follows
    the output crossbar and bounds the classification voltages. This final stage is outside
    the state feedback loop.

    See ``utils.training`` for the training adapter. Device counts cover the modeled
    circuits. This model does not provide SPICE export or total network power.
    """

    def __init__(self, topology, args, cell, target_activation, gate_activation, output_activation,
                 *, assume_ideal_drive, rc_variation=None, constant_gate_input=None,
                 gate_input_share=None, gate_state_share=None):
        super().__init__()
        if len(topology) != 3 or any(type(n) is not int or n < 1 for n in topology):
            raise ValueError("Require one-hidden-layer topology [input, state, output].")
        if assume_ideal_drive is not True:
            raise ValueError("Require explicit assume_ideal_drive=True for unloaded VTC drivers.")
        if not isinstance(cell, GatedStateCell) or any(
                not isinstance(a, NonlinearCircuit)
                for a in (target_activation, gate_activation, output_activation)):
            raise TypeError("Provide the reviewed cell and existing nonlinear circuit instances.")
        self.topology, self.args, self.stage = tuple(topology), args, "train"
        self.state_width = topology[1]
        if constant_gate_input is not None and (
                type(constant_gate_input) not in (int, float) or not math.isfinite(constant_gate_input)):
            raise ValueError("constant_gate_input must be a finite reference voltage.")
        self.constant_gate_input = constant_gate_input
        if not cell.branch.transistor_enabled and constant_gate_input is not None:
            raise ValueError("constant_gate_input requires an enabled transistor branch.")
        self._sequence_active = False
        self._sequence_config = None
        self._rc_sampler = None
        self.rc_variation = None if rc_variation is None else create_variation(rc_variation)
        self.cell = cell
        self.activations = nn.ModuleList([target_activation, gate_activation, output_activation])
        if any(activation.parameter_sets not in (1, topology[1])
               for activation in (*self.activations[:2], cell.readout)):
            raise ValueError("Drive/readout activation parameter sets must be 1 or the state width.")
        if output_activation.parameter_sets not in (1, topology[2]):
            raise ValueError("Output activation parameter sets must be 1 or the class count.")
        self.inverters = nn.ModuleList([
            InverterCircuit(args, variation=create_variation(getattr(args, "circuit_variation", "uniform")))
            for _ in range(3)])
        self.bias_inverter = InverterCircuit(
            args, variation=create_variation(getattr(args, "circuit_variation", "uniform")))
        feature_width = topology[0] + topology[1]
        self.layers = nn.ModuleList([
            pLayer(feature_width, topology[1], args, True,
                   float(a.reference_operating_input().detach()),
                   create_variation(getattr(args, "crossbar_variation", "uniform")))
            for a in self.activations[:2]])
        # The output crossbar is followed by its own VTC so the classification
        # voltages are bounded and bias-initialized like the reference output stage.
        self.layers.append(pLayer(topology[1], topology[2], args, True,
                                  float(output_activation.reference_operating_input().detach()),
                                  create_variation(getattr(args, "crossbar_variation", "uniform"))))
        if gate_input_share is not None:
            # With the default initialization, signal rows carry about 0.4% of each gate column's weight.
            # Give input and feedback rows more weight while preserving the zero-input operating point.
            self.layers[1].crossbar.initialize_operating_sensitivity(
                input_width=topology[0], input_share=gate_input_share,
                state_share=0.0 if gate_state_share is None else gate_state_share,
                operating_input=float(self.activations[1].reference_operating_input().detach()))
        if not self.cell.branch.transistor_enabled:
            # Retain module indices and paired initialization, but the gate path
            # is inactive and must neither execute nor enter the optimizer.
            for module in (self.layers[1], self.activations[1], self.inverters[1]):
                module.requires_grad_(False)
        self.SetParameter("stage", "train")
        _freeze_unselected_temporal_circuits(self)

    def SetParameter(self, name, value):
        """Propagate the network's stage or argument settings to all circuit primitives."""
        if self._sequence_active:
            raise RuntimeError("Cannot change stage/args inside a sequence.")
        if name == "stage":
            if value not in {"train", "valid", "test"}:
                raise ValueError("Unsupported network stage.")
            self.stage = value
        elif name == "args":
            self.args = value
        else:
            raise ValueError("Only stage/args updates are supported.")
        for module in self.modules():
            if isinstance(module, (NonlinearCircuit, Crossbar)):
                module.SetParameter(name, value)

    def _sampling_config(self):
        if self.stage == "test":
            return self.stage, self.args.N_test, self.args.e_test
        return self.stage, self.args.N_train, self.args.e_train

    @contextmanager
    def sequence(self):
        """Reuse the same sampled chips across calls and batches. The caller carries the
        capacitor state.
        """
        if self._sequence_active:
            raise RuntimeError("Nested sequence contexts are not supported.")
        if len({id(a) for a in (*self.activations, self.cell.readout)}) != 4:
            raise ValueError(
                "Target, gate, output and readout need four distinct physical circuit instances.")
        modules = [m for m in self.modules() if isinstance(m, (NonlinearCircuit, Crossbar))]
        originals = [(m, m.variation) for m in modules]
        if any(isinstance(sampler, SequenceVariation) for _, sampler in originals):
            raise RuntimeError("A circuit is already owned by another sequence context.")
        self._sequence_config = self._sampling_config()
        self._sequence_active = True
        self._rc_sampler = None if self.rc_variation is None else SequenceVariation(self.rc_variation)
        try:
            for module, sampler in originals:
                module.variation = SequenceVariation(sampler)
            yield self
        finally:
            for module, sampler in originals:
                module.variation = sampler
            self._sequence_active = False
            self._sequence_config = None
            self._rc_sampler = None

    def GetParam(self, group_str="w,n"):
        """Return trainable crossbar (w) or nonlinear circuit (n) parameters, excluding
        fixed surrogate weights.
        """
        groups = {group.strip().lower() for group in group_str.split(",")}
        parameters = []
        if groups & {"w", "weight"}:
            parameters.extend(layer.crossbar.theta_ for layer in self.layers)
        if groups & {"n", "nonlinear"}:
            parameters.extend(m.rt_ for m in self.modules() if isinstance(m, NonlinearCircuit))
        parameters.extend(
            parameter for parameter in self.cell.branch.parameters()
            if parameter.requires_grad
        )
        return list({id(p): p for p in parameters if p.requires_grad}.values())

    def DeviceInventory(self):
        """Count resistors, transistors, capacitors, and circuit instances in the enabled paths."""
        width = self.state_width
        enabled = self.cell.branch.transistor_enabled
        indices = (0, 1, 2) if enabled else (0, 2)
        roles = ("target", "gate", "output")
        layers = [(roles[i], self.layers[i], self.inverters[i], self.activations[i])
                  for i in indices]
        readout = {kind: quantity * width for kind, quantity in self.cell.readout.device_inventory.items()}
        inventory = _temporal_device_inventory(layers, self.bias_inverter, {
            "state_branch": dict(resistor=width, capacitor=width, transistor=width if enabled else 0),
            "readout": dict(readout, activation_instances=width),
        })
        inventory["state_branch_basis"] = "One R and C per state; one parallel EGT only when enabled."
        inventory["constant_gate_reference_ports"] = int(self.constant_gate_input is not None)
        return inventory

    def DeviceCount(self):
        return self.DeviceInventory()["counts"]

    def forward(self, signal, state, dt, substeps=1, *, trace_samples=0, return_sequence=False):
        """Advance the circuit and return the final state and classification voltages.

        ``signal`` has shape [batch, time, input] and ``state`` has shape [sample, batch,
        state_width]. Both are in volts. Each substep recomputes feedback and gating. The
        caller supplies the initial capacitor state and passes the final capacitor state to
        the next call.

        When ``trace_samples > 0``, record detached CPU traces at each substep for the
        first ``trace_samples`` sequences in the batch.
        """
        if not self._sequence_active:
            with self.sequence():
                return self.forward(signal, state, dt, substeps, trace_samples=trace_samples, return_sequence=return_sequence)
        if type(trace_samples) is not int or trace_samples < 0:
            raise ValueError("trace_samples must be a nonnegative integer.")
        if self._sampling_config() != self._sequence_config:
            raise ValueError("Sampling configuration changed within a sequence.")
        if signal.ndim != 3 or min(signal.shape) < 1 or signal.shape[-1] != self.topology[0]:
            raise ValueError("signal must be nonempty [batch, time, input_width].")
        if not math.isfinite(dt) or dt <= 0 or type(substeps) is not int or substeps < 1:
            raise ValueError("Require finite dt > 0 and integer substeps >= 1.")
        samples = self.args.N_test if self.stage == "test" else self.args.N_train
        epsilon = self.args.e_test if self.stage == "test" else self.args.e_train
        for module in self.modules():
            if isinstance(module, NonlinearCircuit) and module.sampling != (samples, epsilon):
                raise ValueError("All circuits must use the same sampling configuration.")
        if state.shape != (samples, signal.shape[0], self.topology[1]):
            raise ValueError("state must be [sample, batch, state_width].")
        if state.dtype != signal.dtype or state.device != signal.device:
            raise ValueError("State and signal must have matching dtype and device.")
        records, outputs = [], []
        def record_step(values):
            records.append({key: None if value is None else value[:, :trace_samples].detach().cpu().clone()
                            for key, value in values.items()})
        r_scale = c_scale = 1.0
        if self._rc_sampler is not None:
            factors = self._rc_sampler.sample(signal.new_ones(self.topology[1], 2), samples, epsilon)
            r_scale, c_scale = (v.unsqueeze(1) for v in factors.unbind(-1))
        sensed = self.cell.readout(state).output
        bias = self.bias_inverter(torch.ones_like(state[..., :1])).output
        constant_gate = None
        if self.constant_gate_input is not None:
            # Keep the trainable gate circuit, but drive every signal/state input
            # from the same DC reference. Its output is constant for this chip.
            gate_features = state.new_full(
                (samples, signal.shape[0], self.topology[0] + self.state_width), self.constant_gate_input)
            constant_gate = self.layers[1](
                gate_features, self.inverters[1], bias, self.activations[1]).output
        for x in signal.unbind(1):
            x = x.unsqueeze(0).expand(samples, -1, -1)
            for _ in range(substeps):
                features = torch.cat((x, sensed), dim=-1)
                target = self.layers[0](features, self.inverters[0], bias, self.activations[0]).output
                if not self.cell.branch.transistor_enabled:
                    gate = torch.zeros_like(state)  # Interface placeholder; RC-only ignores this voltage.
                else:
                    gate = (self.layers[1](features, self.inverters[1], bias, self.activations[1]).output
                            if constant_gate is None else constant_gate)
                result = self.cell(state, target, gate, dt / substeps, r_scale=r_scale, c_scale=c_scale,
                                   trace=record_step if trace_samples else None)
                state, sensed = result.state, result.output
            if return_sequence:
                outputs.append(self.layers[2](sensed, self.inverters[2], bias,
                                              self.activations[2]).output)
        output = (outputs[-1] if outputs
                  else self.layers[2](sensed, self.inverters[2], bias, self.activations[2]).output)
        trace = None
        if records:
            trace = {key: None if records[0][key] is None else torch.stack([r[key] for r in records], dim=2)
                     for key in ("target", "gate", "conductance", "current", "tau")}
            trace["state"] = torch.stack([records[0]["state_before"]] + [r["state_after"] for r in records], dim=2)
            trace["time"] = torch.arange(len(records)+1, dtype=torch.float64) * (dt/substeps)
        return TemporalNetworkResult(state=state, output=output, trace=trace,
                                     output_sequence=torch.stack(outputs, dim=2) if outputs else None)


class pRNN(nn.Module):
    """Implement the FilterGroup/pRecurrentLayer protocol using the existing circuit
    primitives.

    Each channel has a direct path and ``width`` independent filters. Reading the filter
    states before updating them reproduces the one-step delay in
    ``LearnableFilter.forward``. In ``prnn2``, each filter has two cascaded stages and the
    last stage supplies the readout. Each stage uses the pRNN beta parameterization and
    coupling protocol independently.

    The cascade follows ``SecOLearnableFilter`` in KIT-Neuromorphic-Computing/ADAPT-pNC
    (commit d25c063). This adapter uses nominal circuits and omits that project's physical
    R/C parameterization and augmented training.

    The topology is [input, block_width_1, ..., block_width_L, classes], with the final
    block width equal to the number of classes.
    """

    def __init__(self, topology, args):
        super().__init__()
        self.topology, self.args, self.stage = tuple(topology), args, "train"
        self.widths = tuple(topology[1:-1])
        self.filter_order = 2 if getattr(args, "temporal_model", None) == "prnn2" else 1
        self.state_width = self.filter_order * sum(width * width for width in self.widths)
        self._sequence_active = False
        self.activation = create_activation_circuit(args.activation_circuit, args)
        self.inverter = InverterCircuit(args)
        self.layers = nn.ModuleList()
        self.beta_logits = nn.ParameterList()
        self.second_beta_logits = nn.ParameterList()
        operating_input = float(self.activation.reference_operating_input().detach())
        n_in = topology[0]
        for width in self.widths:
            self.layers.extend([
                pLayer(n_in, width, args, True, operating_input, create_variation(args.crossbar_variation)),
                pLayer(width * (width + 1), width, args, True, operating_input, create_variation(args.crossbar_variation)),
            ])
            beta = 0.001 + torch.arange(width, dtype=torch.float32) / width
            self.beta_logits.append(nn.Parameter(torch.logit(beta).repeat(width, 1)))
            if self.filter_order == 2:
                # ADAPT-pNC's cascade/reversed pairing, using our pRNN coefficient
                # parameterization and initialization grid (not its physical R/C fit).
                self.second_beta_logits.append(nn.Parameter(torch.logit(beta.flip(0)).repeat(width, 1)))
            n_in = width
        self.SetParameter("stage", "train")
        _freeze_unselected_temporal_circuits(self)

    def SetParameter(self, name, value):
        if self._sequence_active:
            raise RuntimeError("Cannot change stage/args inside a sequence.")
        if name == "stage" and value in {"train", "valid", "test"}:
            self.stage = value
        elif name == "args":
            self.args = value
        else:
            raise ValueError("Unsupported pRNN stage/args update.")
        for module in self.modules():
            if isinstance(module, (NonlinearCircuit, Crossbar)):
                module.SetParameter(name, value)

    @contextmanager
    def sequence(self):
        if self._sequence_active:
            raise RuntimeError("Nested sequence contexts are not supported.")
        if self.args.e_train != 0 or self.args.e_test != 0:
            raise ValueError("The pRNN comparison currently supports nominal devices only.")
        self._sequence_active = True
        try:
            yield self
        finally:
            self._sequence_active = False

    def GetParam(self, group_str="w,n"):
        groups = {part.strip().lower() for part in group_str.split(",")}
        parameters = []
        if groups & {"w", "weight"}:
            parameters.extend(layer.crossbar.theta_ for layer in self.layers)
            parameters.extend(self.beta_logits)
            parameters.extend(self.second_beta_logits)
        if groups & {"n", "nonlinear"}:
            parameters.extend((self.activation.rt_, self.inverter.rt_))
        return parameters

    def DeviceInventory(self):
        layers = [(f"block_{i//2}_{'projection' if i % 2 == 0 else 'mixing'}", layer,
                   self.inverter, self.activation) for i, layer in enumerate(self.layers)]
        inventory = _temporal_device_inventory(layers, self.inverter, {
            "filter_bank": dict(resistor=self.state_width, capacitor=self.state_width, transistor=0),
        })
        inventory["state_branch_basis"] = (
            f"{self.filter_order} cascaded R-C states per filter; direct paths add no capacitor. "
            "R/C values are not calibrated from beta.")
        inventory["filter_order"] = self.filter_order
        inventory["constant_gate_reference_ports"] = 0
        return inventory

    def DeviceCount(self):
        return self.DeviceInventory()["counts"]

    def forward(self, signal, state, dt, substeps=1, *, trace_samples=0, return_sequence=False):
        if not self._sequence_active:
            with self.sequence():
                return self.forward(signal, state, dt, substeps, trace_samples=trace_samples, return_sequence=return_sequence)
        if type(trace_samples) is not int or trace_samples < 0:
            raise ValueError("trace_samples must be a nonnegative integer.")
        if substeps != 1 or dt != self.args.recurrent["dt"]:
            raise ValueError("pRNN preserves the original sample-delay update: use configured dt and substeps=1.")
        samples = self.args.N_test if self.stage == "test" else self.args.N_train
        if signal.ndim != 3 or min(signal.shape) < 1 or signal.shape[-1] != self.topology[0]:
            raise ValueError("signal must be nonempty [batch,time,input_width].")
        reference = getattr(self.args, "temporal_protocol", None) == "learnable_filters"
        factor = getattr(self.args, "coupling_factor", None)
        factor = (0.1 if reference else 0.0) if factor is None else factor
        # One coupling per physical filter and forward, held over the whole sequence.
        # Reference order interleaves each filter's random initial state and coupling.
        low, high = initial_state_range(self.args)
        initial, couplings = [], []
        for _ in range(self.state_width):
            if state is None:
                initial.append(low + (high - low) * torch.rand(
                    samples, signal.shape[0], device=signal.device, dtype=signal.dtype))
            couplings.append(1.0 + factor * torch.rand(samples, device=signal.device, dtype=signal.dtype)
                             if factor else signal.new_ones(samples))
        if state is None:
            state = torch.stack(initial, dim=-1)
        coupling = torch.stack(couplings, dim=-1)
        block_sizes = [self.filter_order*w*w for w in self.widths]
        coupling_blocks = [part.reshape(samples, 1, self.filter_order, width, width)
                           for part, width in zip(coupling.split(block_sizes, -1), self.widths)]
        if state.shape != (samples, signal.shape[0], self.state_width):
            raise ValueError("pRNN state must include every channel/filter capacitor.")
        if state.dtype != signal.dtype or state.device != signal.device:
            raise ValueError("State and signal must have matching dtype/device.")
        memories = [part.reshape(samples, signal.shape[0], self.filter_order, width, width)
                    for part, width in zip(state.split(block_sizes, -1), self.widths)]
        bias = self.inverter(torch.ones_like(state[..., :1])).output
        states = [state[:, :trace_samples].detach().cpu().clone()] if trace_samples else []
        outputs = []
        for x in signal.unbind(1):
            output = x.unsqueeze(0).expand(samples, -1, -1)
            next_memories = []
            for index, (memory, beta_logits) in enumerate(zip(memories, self.beta_logits)):
                projected = self.layers[2*index](output, self.inverter, bias, self.activation).output
                # Original per-channel order: direct input, filter 0, ..., filter width-1.
                filtered = torch.cat((projected.unsqueeze(-1), memory[:, :, -1]), -1).flatten(-2)
                output = self.layers[2*index+1](filtered, self.inverter, bias, self.activation).output
                beta = beta_logits.sigmoid() * coupling_blocks[index][:, :, 0]
                first = beta * memory[:, :, 0] + (1-beta) * projected.unsqueeze(-1)
                stages = [first]
                if self.filter_order == 2:
                    beta2 = self.second_beta_logits[index].sigmoid() * coupling_blocks[index][:, :, 1]
                    # The second stage receives the PRE-update first-stage output:
                    # this preserves the two sample delays of cascaded LearnableFilter.
                    stages.append(beta2 * memory[:, :, 1] + (1-beta2) * memory[:, :, 0])
                next_memories.append(torch.stack(stages, dim=2))
            memories = next_memories
            if return_sequence:
                outputs.append(output)
            if trace_samples:
                states.append(torch.cat([m.flatten(-3) for m in memories], -1)[:, :trace_samples].detach().cpu().clone())
        trace = (dict(state=torch.stack(states, dim=2), time=torch.arange(len(states), dtype=torch.float64)*dt,
                      gate=None, conductance=None, current=None, tau=None) if states else None)
        return TemporalNetworkResult(torch.cat([m.flatten(-3) for m in memories], -1), output, trace,
                                     torch.stack(outputs, dim=2) if outputs else None)


class SoftwareRNN(nn.Module):
    """Elman/GRU software baselines with the shared temporal trainer interface.

    Hidden initialization follows the selected temporal protocol. Weights use
    PyTorch defaults; readout is an unconstrained linear layer, not a circuit.
    """

    def __init__(self, topology, args):
        super().__init__()
        self.topology, self.args, self.stage = tuple(topology), args, "train"
        self.widths = tuple(topology[1:-1])
        self.state_width = sum(self.widths)
        self._sequence_active = False
        recurrent = nn.RNN if args.temporal_model == "rnn" else nn.GRU
        self.recurrent_layers = nn.ModuleList(
            recurrent(n_in, width, batch_first=True)
            for n_in, width in zip(topology[:-2], self.widths))
        self.readout = nn.Linear(self.widths[-1], topology[-1])

    def SetParameter(self, name, value):
        if self._sequence_active:
            raise RuntimeError("Cannot change stage/args inside a sequence.")
        if name == "stage" and value in {"train", "valid", "test"}:
            self.stage = value
        elif name == "args":
            self.args = value
        else:
            raise ValueError("Unsupported software baseline stage/args update.")

    @contextmanager
    def sequence(self):
        if self._sequence_active:
            raise RuntimeError("Nested sequence contexts are not supported.")
        self._sequence_active = True
        try:
            yield self
        finally:
            self._sequence_active = False

    def GetParam(self, group_str="w,n"):
        groups = {part.strip().lower() for part in group_str.split(",")}
        return list(self.parameters()) if groups & {"w", "weight"} else []

    def DeviceInventory(self):
        return dict(counts=None, implementation="software", state_branch_basis=
                    "Abstract recurrent states; no printed device or capacitance mapping.")

    def DeviceCount(self):
        return None

    def forward(self, signal, state, dt, substeps=1, *, trace_samples=0, return_sequence=False):
        if not self._sequence_active:
            with self.sequence():
                return self.forward(signal, state, dt, substeps, trace_samples=trace_samples,
                                    return_sequence=return_sequence)
        if type(trace_samples) is not int or trace_samples < 0:
            raise ValueError("trace_samples must be a nonnegative integer.")
        if substeps != 1 or dt != self.args.recurrent["dt"]:
            raise ValueError("Software baselines update once per input sample.")
        if signal.ndim != 3 or min(signal.shape) < 1 or signal.shape[-1] != self.topology[0]:
            raise ValueError("signal must be nonempty [batch,time,input_width].")
        samples = self.args.N_test if self.stage == "test" else self.args.N_train
        expected = (samples, signal.shape[0], self.state_width)
        if state is None or state.shape != expected:
            raise ValueError("Supply every software hidden state, including the layer dimension.")
        if state.dtype != signal.dtype or state.device != signal.device:
            raise ValueError("State and signal must have matching dtype/device.")
        batch, steps = signal.shape[:2]
        output = signal.unsqueeze(0).expand(samples, -1, -1, -1).reshape(samples*batch, steps, -1)
        final, histories = [], []
        for layer, initial in zip(self.recurrent_layers, state.split(self.widths, dim=-1)):
            output, hidden = layer(output, initial.reshape(1, samples*batch, -1).contiguous())
            final.append(hidden.reshape(samples, batch, -1))
            if trace_samples:
                histories.append(output.reshape(samples, batch, steps, -1)[:, :trace_samples])
        logits = self.readout(output).reshape(samples, batch, steps, -1)
        trace = None
        if trace_samples:
            history = torch.cat(histories, dim=-1)
            history = torch.cat((state[:, :trace_samples].unsqueeze(2), history), dim=2)
            trace = dict(state=history.detach().cpu(), time=torch.arange(steps+1, dtype=torch.float64)*dt,
                         gate=None, conductance=None, current=None, tau=None)
        return TemporalNetworkResult(torch.cat(final, dim=-1), logits[:, :, -1], trace,
                                     logits if return_sequence else None)


def temporal_configuration(args):
    """Select the temporal configuration, including for older pLTC checkpoints without
    ``temporal_model``.
    """
    return args.recurrent if getattr(args, "temporal_model", None) in {"prnn", "prnn2", "rnn", "gru"} else args.gated


def temporal_optimizer_groups(model, args):
    """Build optimizer groups with optional learning-rate scaling for the gate and target
    crossbars.

    The gate and target paths in pLTC have the longest gradient chains and can use larger
    initial learning rates. Other parameters keep the base rate. With pRNN or with both
    scale factors set to one, all parameters share one group.
    """
    gate_scale = getattr(args, "gate_lr_scale", 1.0)
    target_scale = getattr(args, "target_lr_scale", 1.0)
    params = list(model.GetParam(args.learnable_group))
    if not isinstance(model, pGatedNN) or (gate_scale == 1.0 and target_scale == 1.0):
        return [{"params": params, "lr": args.LR}]

    boosted = set()
    groups = []
    rest = []
    for parameter in params:
        if parameter is model.layers[1].crossbar.theta_ and gate_scale != 1.0:
            groups.append({"params": [parameter], "lr": args.LR * gate_scale})
            boosted.add(id(parameter))
        elif parameter is model.layers[0].crossbar.theta_ and target_scale != 1.0:
            groups.append({"params": [parameter], "lr": args.LR * target_scale})
            boosted.add(id(parameter))
        else:
            rest.append(parameter)
    if rest:
        groups.insert(0, {"params": rest, "lr": args.LR})
    return groups


def build_temporal_model(args, topology):
    model_name = getattr(args, "temporal_model", None)
    if model_name in (None, "pltc"):
        return build_gated_model(args, topology)
    if model_name not in {"prnn", "prnn2", "rnn", "gru"}:
        raise ValueError("Unknown temporal model.")
    spec = args.recurrent
    required = {"version", "dt", "substeps", "initial_voltage", "valid_seed", "test_seed",
                "assume_ideal_drive"}
    if set(spec) != required or spec["version"] != 1 or spec["assume_ideal_drive"] is not True:
        raise ValueError("Provide the complete version-1 recurrent configuration and ideal-drive assumption.")
    if not math.isfinite(spec["dt"]) or spec["dt"] <= 0 or not math.isfinite(spec["initial_voltage"]):
        raise ValueError("Require positive dt and finite initial_voltage.")
    if type(spec["substeps"]) is not int or spec["substeps"] != 1:
        raise ValueError("pRNN requires substeps=1 to preserve the original discrete delay.")
    if any(type(spec[key]) is not int or not 0 <= spec[key] < 2**63 for key in ("valid_seed", "test_seed")):
        raise ValueError("Evaluation seeds must be integers in [0, 2**63).")
    if (len(topology) < 3 or any(type(n) is not int or n < 1 for n in topology)
            or topology[-1] < 2):
        raise ValueError("Temporal baselines require positive widths and at least two classes.")
    if model_name in {"rnn", "gru"}:
        if (args.MODE != "nominal" or args.powerestimator != "none"
                or args.e_train != 0 or args.e_test != 0
                or getattr(args, "coupling_factor", None) not in (None, 0)):
            raise ValueError("Software baselines require nominal, zero-variation, uncoupled operation.")
        return SoftwareRNN(topology, args).to(args.DEVICE)
    if topology[-2] != topology[-1]:
        raise ValueError("pRNN requires block widths ending at the class count.")
    if (args.circuit_sharing != "network" or args.activation_circuit != "tanh"
            or args.MODE != "nominal" or args.powerestimator != "none"
            or args.e_train != 0 or args.e_test != 0):
        raise ValueError("pRNN requires shared tanh circuits, nominal devices and no power objective.")
    return pRNN(topology, args).to(args.DEVICE)


def build_gated_model(args, topology):
    """Build the float32 pLTC from explicit readout, target, gate, output, branch, and
    timing settings.
    """
    spec = args.gated
    required = {
        "version", "readout", "target", "gate", "resistance", "capacitance",
        "initial_voltage", "dt", "substeps", "rc_variation", "assume_reciprocal",
        "assume_ideal_readout", "assume_ideal_drive", "valid_seed", "test_seed",
    }
    if (set(spec) - {"constant_gate_input", "transistor_enabled", "output",
                     "gate_input_share", "gate_state_share",
                     "learnable_resistance", "resistance_bounds"} != required
            or spec["version"] != 1):
        raise ValueError("Provide the complete version-1 gated configuration.")
    transistor_enabled = spec.get("transistor_enabled", True)
    if type(transistor_enabled) is not bool:
        raise ValueError("transistor_enabled must be a boolean.")
    if not transistor_enabled and "constant_gate_input" in spec:
        raise ValueError("constant_gate_input requires an enabled transistor branch.")
    if "constant_gate_input" in spec and (
            type(spec["constant_gate_input"]) not in (int, float)
            or not math.isfinite(spec["constant_gate_input"])):
        raise ValueError("constant_gate_input must be a finite reference voltage.")
    if spec.get("learnable_resistance", False) not in (True, False):
        raise ValueError("learnable_resistance must be a boolean.")
    if "resistance_bounds" in spec:
        bounds = spec["resistance_bounds"]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(not math.isfinite(value) or value <= 0 for value in bounds)
                or bounds[0] >= bounds[1]):
            raise ValueError("resistance_bounds must contain two finite ordered positive values.")
    if args.powerestimator != "none" or args.MODE != "nominal":
        raise ValueError("Gated training currently requires powerestimator=none and MODE=nominal.")
    if args.circuit_sharing not in {"layer", "neuron"}:
        raise ValueError("Choose layer or neuron sharing; the three VTCs are distinct circuits.")
    if not math.isfinite(spec["initial_voltage"]) or not math.isfinite(spec["dt"]) or spec["dt"] <= 0:
        raise ValueError("Require finite initial_voltage and dt > 0.")
    if type(spec["substeps"]) is not int or spec["substeps"] < 1:
        raise ValueError("Require integer substeps >= 1.")
    if any(type(spec[k]) is not int or not 0 <= spec[k] < 2**63 for k in ("valid_seed", "test_seed")):
        raise ValueError("Evaluation seeds must be integers in [0, 2**63).")
    if len(topology) != 3 or any(type(n) is not int or n < 1 for n in topology) or topology[-1] < 2:
        raise ValueError("Require classification topology [input, state, classes >= 2].")
    parameter_sets = topology[1] if args.circuit_sharing == "neuron" else 1
    output_parameter_sets = topology[2] if args.circuit_sharing == "neuron" else 1
    readout, target, gate = [
        create_activation_circuit(spec[role], args, parameter_sets,
                                  create_variation(args.circuit_variation))
        for role in ("readout", "target", "gate")
    ]
    # The output VTC defaults to the readout circuit type when the configuration
    # predates this stage, so older JSON stays loadable with identical semantics.
    output = create_activation_circuit(spec.get("output", spec["readout"]), args,
                                       output_parameter_sets,
                                       create_variation(args.circuit_variation))
    branch = GatedRCBranch(assume_reciprocal=spec["assume_reciprocal"],
                          resistance=spec["resistance"], capacitance=spec["capacitance"],
                          transistor_enabled=transistor_enabled,
                          learnable_resistance=spec.get("learnable_resistance", False),
                          resistance_bounds=spec.get("resistance_bounds", (1e3, 1e9)),
                          instances=topology[1])
    cell = GatedStateCell(branch, readout, assume_ideal_readout=spec["assume_ideal_readout"])
    return pGatedNN(topology, args, cell, target, gate, output,
                    assume_ideal_drive=spec["assume_ideal_drive"],
                    rc_variation=spec["rc_variation"],
                    constant_gate_input=spec.get("constant_gate_input"),
                    gate_input_share=spec.get("gate_input_share"),
                    gate_state_share=spec.get("gate_state_share")).to(args.DEVICE)
