"""Single-pass evaluators for training and validation."""

import math
from contextlib import contextmanager

import torch

from models.losses import Lossfunction


class Evaluator(torch.nn.Module):
    """Compute loss and performance from the output and power of the same physical forward pass."""

    def __init__(self, args):
        super().__init__()
        self.sensing_margin = 0.01
        self.performance = self.nominal
        self.args = args

    def nominal(self, prediction, label):
        _, index = torch.max(prediction, dim=1)
        return (label.view(-1) == index).float().mean().item()

    def maa(self, prediction, label):
        activation, index = torch.topk(prediction, k=2, dim=1)
        correct = (
            (activation[:, 0] >= self.sensing_margin)
            & (activation[:, 1] <= 0)
            & (label.view(-1) == index[:, 0])
        )
        return correct.float().mean().item()

    def forward(self, network, signal, label):
        if self.args.metric == "acc":
            self.performance = self.nominal
        elif self.args.metric == "maa":
            self.performance = self.maa
        else:
            raise ValueError(f"Unsupported metric: {self.args.metric}.")

        result = network(signal)
        output = result.output
        performance = 0.0
        with torch.no_grad():
            for sample in output:
                performance += self.performance(sample, label)
        return {
            "output": output,
            "accuracy": performance
            / (output.shape[0] + torch.finfo(output.dtype).tiny),
            "power": result.power,
        }

    def evaluate(self, network, signal, label, lossfunction):
        """Compute loss, performance, and input-dependent power in one forward pass."""
        evaluation = self(network, signal, label)
        evaluation["loss"] = lossfunction(
            network,
            evaluation["output"],
            label,
            evaluation["power"],
        )
        return evaluation


class TemporalEvaluator(Evaluator):
    """Evaluate each sequence from its own initial capacitor state. Power estimation is disabled."""

    def evaluate(self, network, signal, label, lossfunction, *, trace_samples=0):
        from models.network import initial_state_range, temporal_configuration
        spec = temporal_configuration(self.args)
        if self.args.powerestimator != "none" or lossfunction.args.powerestimator != "none":
            raise ValueError("Temporal evaluation does not provide a power estimate.")
        samples = self.args.N_test if network.stage == "test" else self.args.N_train
        reference = getattr(self.args, "temporal_protocol", None) == "learnable_filters"
        state = signal.new_full((samples, signal.shape[0], network.state_width), spec["initial_voltage"])
        initial = getattr(self.args, "initial_state", None) or ("uniform" if reference else "constant")
        if initial == "uniform":
            if getattr(self.args, "temporal_model", None) in {"prnn", "prnn2"}:
                # pRNN samples its own per-filter initial states inside the forward pass.
                state = None
            else:
                low, high = initial_state_range(self.args)
                state = low + (high - low) * torch.rand_like(state)
        reduction = getattr(self.args, "temporal_loss", None) or "last"
        readout = getattr(self.args, "temporal_readout", None) or ("sum" if reference else "last")
        if reduction not in {"last", "mean"}:
            raise ValueError("temporal_loss must be last or mean.")
        forward = network(signal, state, spec["dt"], spec["substeps"], trace_samples=trace_samples,
                          return_sequence=reduction == "mean" or readout == "sum")
        output = forward.output
        prediction = forward.output_sequence.sum(dim=2) if readout == "sum" else output
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite temporal classification output.")
        if self.args.metric not in {"acc", "maa"}:
            raise ValueError("Unsupported classification metric.")
        performance = self.nominal if self.args.metric == "acc" else self.maa
        with torch.no_grad():
            accuracy = sum(performance(sample, label) for sample in prediction) / samples
        if reduction == "mean":
            sequence = forward.output_sequence  # [chip, batch, input_time, class]; excludes integration substeps.
            targets = label.reshape(-1, 1).expand(-1, sequence.shape[2]).reshape(-1)
            loss = lossfunction(network, sequence.flatten(1, 2), targets, None)
        else:
            loss = lossfunction(network, output, label, None)
        result = {"loss": loss, "accuracy": accuracy}
        if trace_samples:
            result["trace"] = dict(forward.trace, input=signal[:trace_samples].detach().cpu().clone(),
                                   label=label[:trace_samples].detach().cpu().clone(),
                                   output=output[:, :trace_samples].detach().cpu().clone())
        return result


@contextmanager
def fixed_evaluation_chips(model, seed):
    """Use one set of sampled chips for the entire loader and restore the CPU and CUDA RNG
    states on exit.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        with model.sequence():
            yield


def evaluate_temporal_model(model, loader, *, trace_samples=0, split="test"):
    """Average classification metrics over the sampled chips. All test batches use the
    same set of N_test chips.
    """
    from models.network import temporal_configuration
    if type(trace_samples) is not int or trace_samples < 0:
        raise ValueError("trace_samples must be a nonnegative integer.")
    args = model.args
    model.eval()
    if split not in {"valid", "test"}:
        raise ValueError("Evaluation split must be valid or test.")
    model.SetParameter("stage", split)
    evaluator, lossfunction = TemporalEvaluator(args), Lossfunction(args)
    loss_sum = accuracy_sum = 0.0
    count = 0
    traces = []
    reference = getattr(args, "temporal_protocol", None) == "learnable_filters"
    context = model.sequence() if reference else fixed_evaluation_chips(model, temporal_configuration(args)[split + "_seed"])
    with torch.no_grad(), context:
        for signal, label in loader:
            signal, label = signal.to(args.DEVICE), label.to(args.DEVICE)
            take = min(max(trace_samples-count, 0), label.numel())
            metrics = evaluator.evaluate(model, signal, label, lossfunction, trace_samples=take)
            if take:
                traces.append(dict(metrics["trace"], sample_index=torch.arange(count, count+take)))
            loss = metrics["loss"].item()
            if not math.isfinite(loss) or not math.isfinite(metrics["accuracy"]):
                raise FloatingPointError("Non-finite temporal test metrics.")
            count += label.numel()
            loss_sum += loss * label.numel()
            accuracy_sum += metrics["accuracy"] * label.numel()
    if count == 0:
        raise ValueError("Test DataLoader yielded no samples.")
    result = {"loss": loss_sum / count, "accuracy": accuracy_sum / count}
    if trace_samples:
        result["trace"] = traces
    return result


GatedEvaluator = TemporalEvaluator
evaluate_gated_model = evaluate_temporal_model
