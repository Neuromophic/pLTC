"""Training checkpoints that store model state dictionaries."""

import os
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def load_checkpoint(path):
    """Load a checkpoint and validate its format."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, pickle.UnpicklingError, RuntimeError) as error:
        raise RuntimeError(f"Checkpoint is corrupt: {path}.") from error
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(f"Unsupported checkpoint format: {path}.")
    return payload


def load_completed_checkpoint(path):
    """Load a completed checkpoint for standalone evaluation."""
    payload = load_checkpoint(path)
    if payload.get("status") != "completed":
        raise ValueError(f"Checkpoint is not completed: {path}.")
    expected_types = {
        "config": dict,
        "topology": list,
        "model_state": dict,
        "runtime_state": dict,
    }
    invalid = [
        key
        for key, value_type in expected_types.items()
        if not isinstance(payload.get(key), value_type)
    ]
    if invalid:
        raise ValueError(f"Completed checkpoint is missing required fields: {invalid}.")
    return payload


def _capture_random_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _runtime_state(model):
    getter = getattr(model, "checkpoint_runtime_state", None)
    return {} if getter is None else getter()


def _restore_runtime_state(model, state):
    if not state:
        return
    loader = getattr(model, "load_checkpoint_runtime_state", None)
    if loader is None:
        raise ValueError(
            "Checkpoint contains runtime_state, but the model cannot restore it."
        )
    loader(state)


class CheckpointStore:
    """Manage the latest, best, and completed checkpoints for one experiment."""

    def __init__(self, directory, run_id):
        self.directory = Path(directory)
        self.run_id = run_id

    @property
    def latest_path(self):
        return Path(os.path.join(os.fspath(self.directory), f"{self.run_id}.last.ckp"))

    @property
    def best_path(self):
        return Path(os.path.join(os.fspath(self.directory), f"{self.run_id}.best.ckp"))

    def _payload(
        self,
        model,
        optimizer,
        scheduler,
        progress,
        config,
        topology,
        status,
        pruning=None,
    ):
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "status": status,
            "model_state": model.state_dict(),
            "runtime_state": _runtime_state(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "random_state": _capture_random_state(),
            "progress": dict(progress),
            "config": dict(config),
            "topology": list(topology),
        }
        if pruning is not None:
            payload["pruning"] = dict(pruning)
        return payload

    def _save(
        self,
        path,
        model,
        optimizer,
        scheduler,
        progress,
        config,
        topology,
        status,
        pruning=None,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(
                    self._payload(
                        model,
                        optimizer,
                        scheduler,
                        progress,
                        config,
                        topology,
                        status,
                        pruning,
                    ),
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if temporary_path.is_file():
                temporary_path.unlink()
            raise

    def save_latest(self, model, optimizer, scheduler, progress, config, topology):
        status = progress.get("status", "running")
        if status not in {"running", "completed", "timed_out", "failed"}:
            raise ValueError(f"Latest checkpoint has invalid status: {status}.")
        self._save(
            self.latest_path,
            model,
            optimizer,
            scheduler,
            progress,
            config,
            topology,
            status,
        )

    def save_best(self, model, optimizer, scheduler, progress, config, topology):
        self._save(
            self.best_path,
            model,
            optimizer,
            scheduler,
            progress,
            config,
            topology,
            "running",
        )

    def save_completed(
        self,
        path,
        model,
        optimizer,
        scheduler,
        progress,
        config,
        topology,
        pruning=None,
    ):
        self._save(
            Path(path),
            model,
            optimizer,
            scheduler,
            progress,
            config,
            topology,
            "completed",
            pruning,
        )

    def _load(self, path):
        return load_checkpoint(path)

    def _restore(self, path, model, optimizer, scheduler, config, topology):
        payload = self._load(path)
        self._validate_context(payload, path, config, topology)
        model.load_state_dict(payload["model_state"])
        _restore_runtime_state(model, payload["runtime_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None:
            if payload["scheduler_state"] is None:
                raise ValueError(f"Checkpoint is missing scheduler state: {path}.")
            scheduler.load_state_dict(payload["scheduler_state"])
        _restore_random_state(payload["random_state"])
        return payload["progress"]

    @staticmethod
    def _validate_context(payload, path, config, topology):
        if payload["config"] != dict(config):
            raise ValueError(f"Checkpoint configuration does not match: {path}.")
        if payload["topology"] != list(topology):
            raise ValueError(f"Checkpoint topology does not match: {path}.")

    def restore_latest(self, model, optimizer, scheduler, config, topology):
        if not self.latest_path.is_file():
            raise FileNotFoundError(
                f"Latest checkpoint does not exist: {self.latest_path}."
            )
        return self._restore(
            self.latest_path, model, optimizer, scheduler, config, topology
        )

    def restore_best(self, model, optimizer, scheduler, config, topology):
        if not self.best_path.is_file():
            raise FileNotFoundError(
                f"Best checkpoint does not exist: {self.best_path}."
            )
        return self._restore(
            self.best_path, model, optimizer, scheduler, config, topology
        )

    def restore_best_model(self, model, config, topology):
        if not self.best_path.is_file():
            raise FileNotFoundError(
                f"Best checkpoint does not exist: {self.best_path}."
            )
        payload = self._load(self.best_path)
        self._validate_context(payload, self.best_path, config, topology)
        model.load_state_dict(payload["model_state"])
        _restore_runtime_state(model, payload["runtime_state"])
        return payload["progress"]

    def save_record(
        self, epoch, model, optimizer, scheduler, progress, config, topology
    ):
        path = Path(
            os.path.join(
                os.fspath(self.directory), f"{self.run_id}.epoch-{epoch:06d}.ckp"
            )
        )
        self._save(
            path, model, optimizer, scheduler, progress, config, topology, "running"
        )

    def has_latest(self):
        return self.latest_path.is_file()

    def has_best(self):
        return self.best_path.is_file()

    def is_completed(self, path):
        path = Path(path)
        if not path.is_file():
            return False
        payload = self._load(path)
        return payload["status"] == "completed"
