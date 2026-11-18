"""Run IDs and standalone checkpoints for pruned models."""

from pathlib import Path

from . import FormatUUIDValue
from .checkpoint import CheckpointStore


def pruning_run_id(run_id, mode, value, scope=None):
    """Build a stable, Windows-safe model ID from the requested pruning settings."""
    formatted_value = FormatUUIDValue(float(value))
    if mode == "threshold":
        if scope is not None:
            raise ValueError("Threshold pruning does not accept a scope.")
        return f"{run_id}_pruned-threshold-{formatted_value}"
    if mode == "ratio":
        if scope not in {"network", "layer"}:
            raise ValueError("Ratio pruning requires scope 'network' or 'layer'.")
        return f"{run_id}_pruned-{scope}-ratio-{formatted_value}"
    raise ValueError(f"Unsupported pruning mode: {mode}.")


def save_pruned_model(
    model,
    mode,
    value,
    directory,
    run_id,
    optimizer,
    scheduler,
    progress,
    config,
    topology,
    scope=None,
):
    """Prune the model and atomically replace the completed checkpoint for these pruning
    settings.
    """
    report = model.Prune(mode=mode, value=value, scope=scope)
    pruned_run_id = pruning_run_id(run_id, mode, value, scope)
    path = Path(directory) / f"pNN_{pruned_run_id}"
    CheckpointStore(directory, pruned_run_id).save_completed(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        config=config,
        topology=topology,
        pruning=report,
    )
    return path, report
