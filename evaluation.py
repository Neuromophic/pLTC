"""Evaluate a completed training checkpoint on the test set."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import pNN
from models.network import build_temporal_model, temporal_configuration
from utils import CheckDevice, GetDataLoader, SetSeed, CodeFingerprint, ValidateRuntimeConfiguration
from utils.checkpoint import load_completed_checkpoint
from utils.evaluation import Evaluator, evaluate_temporal_model


EVALUATION_FORMAT_VERSION = 1
EVALUATION_KEYS = (
    "metric",
    "N_test",
    "e_test",
    "R_test",
    "IN_test",
    "M_test",
    "K_test",
    "t_test_max",
    "integration",
    "MODE",
)
REPLACE_RETRY_COUNT = 20
REPLACE_RETRY_DELAY_SECONDS = 0.1


def _native_file_path(path):
    """Add the Windows extended-length path prefix for file I/O."""
    value = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def _result_path_components(run_id):
    """Use a short result directory and store the full run ID and configuration in the JSON
    record.
    """
    return ["eval-" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:20]]


def evaluation_parser():
    """Build the argument parser for standalone evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a completed pNN checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="completed model checkpoint")
    source.add_argument("--projectname", type=Path, help="project directory containing models/")
    parser.add_argument("--DATASET", type=int, help="dataset ID stored during training")
    parser.add_argument("--SEED", type=int, help="training seed")
    parser.add_argument(
        "--resultpath",
        type=Path,
        default=None,
        help="directory for atomic evaluation result files",
    )
    parser.add_argument(
        "--metric", type=str.lower, choices=("acc", "maa"), default=None
    )
    parser.add_argument("--N_test", type=int, default=None)
    parser.add_argument("--trace-samples", type=int, default=0,
                        help="record temporal trajectories for the first N test samples (default: off)")
    parser.add_argument("--e_test", type=float, default=None)
    parser.add_argument("--R_test", type=int, default=None)
    parser.add_argument("--IN_test", type=float, default=None)
    parser.add_argument("--M_test", type=int, default=None)
    parser.add_argument("--K_test", type=int, default=None)
    parser.add_argument("--t_test_max", type=int, default=None)
    parser.add_argument(
        "--integration", type=str.lower, choices=("mc", "gq"), default=None
    )
    parser.add_argument(
        "--MODE", type=str.lower, choices=("nominal", "aging"), default=None
    )
    parser.add_argument("--DataPath", default="dataset")
    return parser


def evaluation_configuration(training_config, overrides):
    """Merge the training configuration with explicit evaluation overrides."""
    settings = {}
    for key in EVALUATION_KEYS:
        value = overrides.get(key)
        if value is None:
            if key not in training_config:
                raise ValueError(
                    f"Checkpoint configuration is missing evaluation key: {key}."
                )
            value = training_config[key]
        settings[key] = value
    trace_samples = overrides.get("trace_samples", 0)
    if type(trace_samples) is not int or trace_samples < 0:
        raise ValueError("trace_samples must be a nonnegative integer.")
    if trace_samples:
        if training_config.get("task") != "temporal":
            raise ValueError("Trajectory recording requires a temporal checkpoint.")
        settings["trace_samples"] = trace_samples
    return settings


def evaluation_uuid(training_uuid, settings):
    """Build a stable, Windows-safe ID for the model and its evaluation settings."""
    identity = json.dumps(dict(training_uuid=training_uuid, settings=settings),
                          sort_keys=True, allow_nan=False)
    return "evaluation-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def completed_result_path(model_path, run_id, resultpath=None):
    """Return the path to the result file that marks the evaluation as complete."""
    model_path = Path(model_path)
    directory = (
        Path(os.path.join(os.fspath(model_path.parent.parent), "evaluation"))
        if resultpath is None
        else Path(resultpath)
    )
    return Path(directory, *_result_path_components(run_id), "result.txt")


def _read_result(path):
    try:
        with open(_native_file_path(path), encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Evaluation result is corrupt: {path}.") from error
    if (
        not isinstance(result, dict)
        or result.get("format_version") != EVALUATION_FORMAT_VERSION
    ):
        raise ValueError(f"Unsupported evaluation result format: {path}.")
    return result


def write_completed_result(path, result):
    """Write a completed result atomically so an interruption cannot leave a partial JSON file."""
    path = Path(path)
    os.makedirs(_native_file_path(path.parent), exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".evaluation.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(REPLACE_RETRY_COUNT):
            try:
                os.replace(temporary_path, _native_file_path(path))
                return
            except PermissionError:
                if attempt == REPLACE_RETRY_COUNT - 1:
                    raise
                time.sleep(REPLACE_RETRY_DELAY_SECONDS)
    except Exception:
        if temporary_path.is_file():
            temporary_path.unlink()
        raise


def _scalar_metrics(evaluation):
    metrics = {}
    for key, value in evaluation.items():
        if key == "output":
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(f"Evaluator metric {key!r} must be scalar.")
            metrics[key] = value.detach().item()
        elif isinstance(value, (int, float)):
            metrics[key] = value
        else:
            raise TypeError(
                f"Evaluator metric {key!r} has unsupported type: {type(value).__name__}."
            )
    return metrics


def evaluate_test_loader(model, loader, args, lossfunction):
    """Aggregate scalar metrics over the test set using the test-stage device sampling settings."""
    model.eval()
    model.SetParameter("stage", "test")
    evaluator = Evaluator(args)
    totals = {}
    item_count = 0
    with torch.no_grad():
        for signal, label in loader:
            signal = signal.to(args.DEVICE, non_blocking=True)
            label = label.to(args.DEVICE, non_blocking=True)
            metrics = _scalar_metrics(
                evaluator.evaluate(model, signal, label, lossfunction)
            )
            batch_size = label.numel()
            item_count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
    if item_count == 0:
        raise ValueError("Test DataLoader yielded no samples.")
    return {key: value / item_count for key, value in totals.items()}


def load_temporal_model(path, device="cpu"):
    """Load the completed temporal checkpoint selected by validation loss for this adapter."""
    payload = load_completed_checkpoint(path)
    if payload["progress"].get("selection") != "best_valid_loss":
        raise ValueError("Expected a completed temporal model selected by validation loss.")
    args = SimpleNamespace(**dict(payload["config"], DEVICE=str(device)))
    with torch.random.fork_rng():
        model = build_temporal_model(args, payload["topology"])
    model.load_state_dict(payload["model_state"])
    return model, payload


load_gated_model = load_temporal_model


def _restore_model(model_path, settings):
    payload = load_completed_checkpoint(model_path)
    configuration = dict(payload["config"])
    configuration.update(settings)
    args = SimpleNamespace(**configuration)
    args.DEVICE = CheckDevice(args)
    if configuration.get("task") == "temporal":
        ValidateRuntimeConfiguration(args)
        if payload["progress"].get("selection") != "best_valid_loss":
            raise ValueError("Temporal evaluation requires the validation-best completed export.")
        if args.source_sha256 != CodeFingerprint():
            raise ValueError("Source/assets differ from the temporal training checkpoint.")
        if args.MODE != "nominal" or args.IN_test != 0:
            raise ValueError("Temporal evaluation requires nominal mode and zero input noise.")
        model = build_temporal_model(args, payload["topology"])
        model.load_state_dict(payload["model_state"])
        return payload, args, model
    model = pNN.pNN(payload["topology"], args).to(args.DEVICE)
    model.load_state_dict(payload["model_state"])
    model.load_checkpoint_runtime_state(payload["runtime_state"])
    return payload, args, model


def run_evaluation(command):
    """Run the evaluation or reuse the completed result for the same settings."""
    if command.model is None:
        raise ValueError("Provide a completed checkpoint.")
    model_path = Path(command.model).resolve()
    payload = load_completed_checkpoint(model_path)
    settings = evaluation_configuration(payload["config"], vars(command))
    training_uuid = model_path.name.removeprefix("pNN_")
    run_id = evaluation_uuid(training_uuid, settings)
    result_path = completed_result_path(model_path, run_id, command.resultpath)
    temporal = payload["config"].get("task") == "temporal"
    # Verify source/data before trusting a cached temporal evaluation result.
    if temporal:
        payload, args, model = _restore_model(model_path, settings)
        test_loader, data_info = GetDataLoader(args, "test")
    if os.path.isfile(_native_file_path(result_path)):
        existing = _read_result(result_path)
        if (
            existing.get("status") == "completed"
            and existing.get("evaluation_uuid") == run_id
        ):
            print(f"Evaluation already completed for UUID: {run_id}. Skipping.")
            return result_path, existing
        raise RuntimeError(
            f"Evaluation result exists but is not completed: {result_path}."
        )

    if not temporal:
        payload, args, model = _restore_model(model_path, settings)
    SetSeed(payload["config"]["SEED"])
    if not temporal:
        test_loader, data_info = GetDataLoader(args, "test")
    lossfunction = pNN.Lossfunction(args).to(args.DEVICE)
    validation_metrics = None
    if temporal and getattr(args, "temporal_protocol", None) == "learnable_filters":
        valid_loader, _ = GetDataLoader(args, "valid")
        # Public notebook resets the run seed, evaluates validation, then test.
        validation_metrics = evaluate_temporal_model(model, valid_loader, split="valid")
    metrics = (evaluate_temporal_model(model, test_loader, trace_samples=settings.get("trace_samples", 0)) if temporal else
               evaluate_test_loader(model, test_loader, args, lossfunction))
    traces = metrics.pop("trace", None)
    result = {
        "format_version": EVALUATION_FORMAT_VERSION,
        "status": "completed",
        "training_uuid": training_uuid,
        "evaluation_uuid": run_id,
        "model_path": str(model_path),
        "dataset": data_info["dataname"],
        "training_config": payload["config"],
        "evaluation_config": settings,
        "metrics": metrics,
    }
    if temporal:
        if validation_metrics is not None:
            result["validation_metrics"] = validation_metrics
            result["selection_rule"] = "validation_accuracy_desc_then_validation_loss_then_seed"
        result["temporal_data"] = data_info["metadata"]
        model_name = getattr(args, "temporal_model", None) or "pltc"
        result["temporal_model"] = model_name
        result["state_count"] = model.state_width
        software = model_name in {"rnn", "gru"}
        result["gated_config" if model_name == "pltc" else "recurrent_config"] = temporal_configuration(args)
        result["source_sha256"] = args.source_sha256
        result["device_inventory"] = model.DeviceInventory()
        result["device_count"] = result["device_inventory"]["counts"]
        result["trainable_parameter_count"] = sum(p.numel() for p in model.GetParam() if p.requires_grad)
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        result["implementation"] = "software" if software else "printed_circuit_model"
        result["state_kind"] = "hidden_state" if software else "capacitor_voltage"
        result["filter_order"] = getattr(model, "filter_order", None)
        result["total_capacitance_f"] = (model.state_width * model.cell.branch.capacitance.item()
                                         if model_name == "pltc" else None)
        if traces is not None:
            result["trace"] = {
                "requested_samples": settings["trace_samples"],
                "recorded_samples": sum(len(batch["sample_index"]) for batch in traces),
                "sample_order": "test loader order; indices are zero-based ordinals",
                "axes": {"state": ["chip", "sample", "boundary", "state"],
                         "branch": ["chip", "sample", "substep", "state"],
                         "input": ["sample", "input_step", "feature"],
                         "output": ["chip", "sample", "class"], "time": ["boundary"]},
                "units": {"state": "V", "target": "V", "gate": "V", "input": "V", "output": "V",
                          "conductance": "S", "current": "A", "tau": "s", "time": "s"},
                "semantics": ("pRNN filter states include initial and post-update states. Output uses pre-update "
                              "memories at the last input step. Gate, conductance, current and tau are unavailable."
                              if model_name in {"prnn", "prnn2"} else
                              "State includes initial and post-substep voltages. Branch quantities are sampled "
                              "at each substep start. Conductance is total secant G; current is G*(target-state); "
                              "tau is local C/G, not whole-network memory. RC-only gate is null."),
                "batches": [{key: value.tolist() if isinstance(value, torch.Tensor) else value
                             for key, value in batch.items()} for batch in traces],
            }
            if model_name == "prnn2":
                result["trace"]["semantics"] += (
                    " Two stages per filter; each receives its predecessor's PRE-update output. "
                    "State order: block, stage, channel, filter.")
            if software:
                result["trace"]["semantics"] = (
                    "Software hidden states in layer order, including initial and post-sample states. "
                    "Outputs are linear logits; no device currents, conductances or capacitances.")
                for key in ("state", "target", "gate", "input", "output"):
                    result["trace"]["units"][key] = "dimensionless"
    else:
        result["device_count"] = model.DeviceCount()
    write_completed_result(result_path, result)
    print(f"Evaluation completed: {result_path}.")
    return result_path, result


def completed_model(project, dataset, seed):
    matches = []
    fingerprint = CodeFingerprint()
    for path in sorted((project / "models").glob(f"pNN_*_seed-{seed}_*")):
        config = load_completed_checkpoint(path)["config"]
        if config["DATASET"] == dataset and config["SEED"] == seed:
            if config.get("task") != "temporal" or config.get("source_sha256") == fingerprint:
                matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Found {len(matches)} matching checkpoints in {project} for dataset {dataset}, "
                         f"seed {seed}. Use --model to select a checkpoint explicitly.")
    return matches[0]


def main():
    parser = evaluation_parser()
    args = parser.parse_args()
    if args.projectname is not None:
        if args.DATASET is None or args.SEED is None:
            parser.error("--projectname requires --DATASET and --SEED")
        args.model = completed_model(args.projectname, args.DATASET, args.SEED)
        if args.resultpath is None:
            args.resultpath = args.projectname / "eval"
    run_evaluation(args)


if __name__ == "__main__":
    main()
