"""Training loop and state transitions for pNN models."""

import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import CheckpointStore
from .evaluation import Evaluator, TemporalEvaluator, fixed_evaluation_chips
from models.losses import Lossfunction
from models.network import build_temporal_model, temporal_configuration, temporal_optimizer_groups


def _move_batch(signal, label, device):
    return signal.to(device, non_blocking=True), label.to(device, non_blocking=True)


def _check_finite(value, name, phase, epoch):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {phase} {name} at epoch {epoch}.")


def _check_model_gradients(model, epoch):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                f"Non-finite training gradient for parameter {name} at epoch {epoch}."
            )


def _check_model_parameters(model, epoch):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(
                f"Non-finite training parameter {name} after optimizer step at epoch {epoch}."
            )


def _serialise_value(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialise_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialise_value(item) for key, item in value.items()}
    raise TypeError(
        f"Unsupported checkpoint configuration value: {type(value).__name__}."
    )


def serialise_configuration(args):
    """Convert the experiment configuration to a dictionary of plain values for comparison
    and storage.
    """
    return {key: _serialise_value(value) for key, value in sorted(vars(args).items())
            if not (key in {"gated", "voltage_range", "recurrent", "temporal_model", "temporal_loss", "temporal_encoding",
                            "temporal_protocol", "temporal_readout", "initial_state", "initial_range",
                            "coupling_factor", "sweep_config", "sweep_evaluate",
                            "sweep_trace_samples"} and value is None)}


@dataclass
class TrainingResult:
    """Training status, final epoch, and best validation loss."""

    status: str
    epoch: int
    best_valid_loss: float

    @property
    def completed(self):
        return self.status == "completed"


class ProgressiveScheduler:
    """Reduce the learning rate when validation stalls. Save and restore the scheduler state
    with the checkpoint.
    """

    def __init__(self, optimizer, patience, decay, minimum):
        self.optimizer = optimizer
        self.patience = patience
        self.decay = decay
        self.minimum = minimum
        self.bad_epochs = 0

    def step(self, improved):
        if improved:
            self.bad_epochs = 0
            return False, False
        self.bad_epochs += 1
        if self.bad_epochs <= self.patience:
            return False, False
        self.bad_epochs = 0
        for group in self.optimizer.param_groups:
            group["lr"] *= self.decay
        reached_minimum = all(
            group["lr"] < self.minimum for group in self.optimizer.param_groups
        )
        return True, reached_minimum

    def state_dict(self):
        return {
            "patience": self.patience,
            "decay": self.decay,
            "minimum": self.minimum,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state):
        expected = {"patience", "decay", "minimum", "bad_epochs"}
        if set(state) != expected:
            raise ValueError("Invalid progressive scheduler state.")
        self.patience = state["patience"]
        self.decay = state["decay"]
        self.minimum = state["minimum"]
        self.bad_epochs = state["bad_epochs"]


class Trainer:
    """Coordinate training, validation, checkpointing, early stopping, and learning-rate
    scheduling.
    """

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        lossfunction,
        optimizer,
        args,
        logger,
        run_id,
        topology,
        final_path=None,
        progressive=None,
        resume_from="latest",
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lossfunction = lossfunction
        self.optimizer = optimizer
        self.args = args
        self.logger = logger
        self.run_id = run_id
        self.topology = list(topology)
        self.config = serialise_configuration(args)
        self.store = CheckpointStore(args.temppath, run_id)
        self.record_store = (
            CheckpointStore(args.recordpath, run_id)
            if getattr(args, "recording", False)
            else None
        )
        self.final_path = None if final_path is None else Path(final_path)
        if resume_from not in {"latest", "best"}:
            raise ValueError(f"Unsupported resume source: {resume_from}.")
        self.resume_from = resume_from
        use_progressive = (
            getattr(args, "PROGRESSIVE", False) if progressive is None else progressive
        )
        self.scheduler = (
            ProgressiveScheduler(
                optimizer, args.LR_PATIENCE, args.LR_DECAY, args.LR_MIN
            )
            if use_progressive
            else None
        )
        self.evaluator = Evaluator(args)

    def _log(self, level, message):
        method = getattr(self.logger, level, None)
        if method is not None:
            method(message)
        print(message)

    def _set_stage(self, stage):
        self.model.SetParameter("stage", stage)

    def _epoch_context(self, training):
        """Let temporal trainers set the lifetime of sampled chips during validation."""
        return nullcontext()

    def _run_epoch(self, loader, epoch, training):
        if training:
            set_epoch = getattr(getattr(loader, "dataset", None), "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            if getattr(loader, "generator", None) is not None:
                # Reseed the RandomSampler each epoch for distinct, reproducible shuffle orders.
                loader.generator.manual_seed(getattr(self.args, "SEED", 0) + epoch)

        if training:
            self.model.train()
            self._set_stage("train")
        else:
            self.model.eval()
            self._set_stage("valid")

        loss_sum = accuracy_sum = power_sum = 0.0
        has_power = None
        item_count = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context, self._epoch_context(training):
            for signal, label in loader:
                signal, label = _move_batch(signal, label, self.args.DEVICE)
                evaluation = self.evaluator.evaluate(
                    self.model, signal, label, self.lossfunction
                )
                loss = evaluation["loss"]
                accuracy = evaluation["accuracy"]
                if (getattr(self.args, "temporal_protocol", None) == "learnable_filters"
                        and getattr(self.args, "train_accuracy_forward", False)):
                    # The public trainer computes the training accuracy with a second
                    # independent stochastic forward. It only feeds the log line, so it
                    # is opt-in: skipping it saves about a fifth of the epoch time and
                    # only changes which random draws the stochastic protocol consumes.
                    with torch.no_grad():
                        accuracy = self.evaluator.evaluate(self.model, signal, label, self.lossfunction)["accuracy"]
                power = evaluation.get("power")
                phase = "training" if training else "validation"
                _check_finite(loss, "loss", phase, epoch)
                if has_power is not None and has_power != (power is not None):
                    raise ValueError("Power availability changed within an epoch.")
                has_power = power is not None
                if has_power:
                    _check_finite(power, "power", phase, epoch)
                    if torch.any(power < 0):
                        raise FloatingPointError(
                            f"Negative {phase} power at epoch {epoch}."
                        )
                if not math.isfinite(accuracy):
                    raise FloatingPointError(
                        f"Non-finite {phase} accuracy at epoch {epoch}."
                    )
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    _check_model_gradients(self.model, epoch)
                    self.optimizer.step()
                    _check_model_parameters(self.model, epoch)
                batch_items = label.numel()
                item_count += batch_items
                loss_sum += loss.detach().item() * batch_items
                accuracy_sum += accuracy * batch_items
                if has_power:
                    power_sum += power.detach().item() * batch_items

        if item_count == 0:
            raise ValueError("DataLoader yielded no samples.")
        return {
            "loss": loss_sum / (item_count + torch.finfo(torch.float64).tiny),
            "accuracy": accuracy_sum / (item_count + torch.finfo(torch.float64).tiny),
            "power": power_sum / item_count if has_power else None,
        }

    def _save_latest(self, progress):
        self.store.save_latest(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            progress=progress,
            config=self.config,
            topology=self.topology,
        )

    def _save_best(self, progress):
        self.store.save_best(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            progress=progress,
            config=self.config,
            topology=self.topology,
        )

    def _save_record(self, epoch, progress):
        if self.record_store is not None:
            self.record_store.save_record(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                progress=progress,
                config=self.config,
                topology=self.topology,
            )

    def _complete(self, progress):
        progress = dict(progress, status="completed")
        self._save_latest(progress)
        if self.final_path is not None:
            self.store.save_completed(
                path=self.final_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                progress=progress,
                config=self.config,
                topology=self.topology,
            )
        return TrainingResult(
            "completed", progress["epoch"], progress["best_valid_loss"]
        )

    def _resume_or_start(self):
        progress = {
            "epoch": -1,
            "best_valid_loss": math.inf,
            "patience": 0,
            "status": "running",
        }
        has_checkpoint = (
            self.store.has_latest()
            if self.resume_from == "latest"
            else self.store.has_best()
        )
        if not has_checkpoint:
            return progress
        restore = (
            self.store.restore_latest
            if self.resume_from == "latest"
            else self.store.restore_best
        )
        progress = restore(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            topology=self.topology,
        )
        if progress["status"] == "failed":
            raise RuntimeError(
                "Previous training failed; inspect the latest checkpoint before resuming."
            )
        self._log("info", f"Resume training from epoch {progress['epoch'] + 1}.")
        return progress

    def fit(self, max_hours=None):
        start_time = time.time()
        time_limit = self.args.TIMELIMITATION if max_hours is None else min(self.args.TIMELIMITATION, max_hours)
        progress = self._resume_or_start()
        if progress["status"] == "completed":
            if self.final_path is not None and not self.store.is_completed(
                self.final_path
            ):
                return self._complete(progress)
            return TrainingResult(
                "completed", progress["epoch"], progress["best_valid_loss"]
            )

        try:
            for epoch in range(progress["epoch"] + 1, self.args.EPOCH):
                if time.time() - start_time >= time_limit * 3600:
                    progress.update(status="timed_out")
                    self._save_latest(progress)
                    return TrainingResult(
                        "timed_out", progress["epoch"], progress["best_valid_loss"]
                    )

                epoch_start = time.time()
                train = self._run_epoch(self.train_loader, epoch, training=True)
                valid = self._run_epoch(self.valid_loader, epoch, training=False)
                self.model.train()
                self._set_stage("train")

                improved = valid["loss"] < progress["best_valid_loss"]
                if improved:
                    progress["best_valid_loss"] = valid["loss"]
                    progress["best_valid_accuracy"] = valid["accuracy"]
                    progress["patience"] = 0
                else:
                    progress["patience"] += 1
                progress.update(epoch=epoch, status="running")

                if improved:
                    self._save_best(progress)

                lr_changed = reached_minimum = False
                if self.scheduler is not None:
                    lr_changed, reached_minimum = self.scheduler.step(improved)
                    if lr_changed:
                        progress["patience"] = 0
                        if not reached_minimum:
                            self.store.restore_best_model(
                                self.model, self.config, self.topology
                            )
                            if getattr(self.args, "temporal_protocol", None) == "learnable_filters":
                                # Reference code replaces parameter objects with
                                # those from the best model: Adam starts fresh.
                                self.optimizer.state.clear()
                            self._log(
                                "info",
                                f"Restore best model and reduce learning rate to {self.optimizer.param_groups[0]['lr']}.",
                            )

                self._save_record(epoch, progress)

                if reached_minimum or (
                    self.scheduler is None and progress["patience"] > self.args.PATIENCE
                ):
                    return self._complete(progress)
                if time.time() - start_time >= time_limit * 3600:
                    progress["status"] = "timed_out"
                    self._save_latest(progress)
                    return TrainingResult(
                        "timed_out", epoch, progress["best_valid_loss"]
                    )

                self._save_latest(progress)
                power_text = "" if train["power"] is None else f" Power: {train['power']:.2e} |"
                self._log(
                    "info",
                    f"| Epoch: {epoch:-6d} | Train loss: {train['loss']:.4f} | Valid loss: {valid['loss']:.4f} "
                    f"| Train acc: {train['accuracy']:.4f} | Valid acc: {valid['accuracy']:.4f} "
                    f"| patience: {progress['patience']:-3d} | lr: {self.optimizer.param_groups[0]['lr']:.3e} "
                    f"| Epoch time: {time.time() - epoch_start:.1f} |{power_text}",
                )

            return self._complete(progress)
        except Exception as error:
            progress.update(status="failed", error=f"{type(error).__name__}: {error}")
            self._save_latest(progress)
            self._log("warning", f"Training failed: {progress['error']}")
            raise


class TemporalTrainer(Trainer):
    """Train and validate on voltage tensors shaped [batch, time, input]. The test split is
    reserved for evaluation.
    """

    def __init__(self, train_loader, valid_loader, args, run_id, topology,
                 final_path=None, logger=None):
        if args.EPOCH < 1:
            raise ValueError("Require EPOCH >= 1.")
        model = build_temporal_model(args, topology)
        optimizer = torch.optim.Adam(temporal_optimizer_groups(model, args))
        super().__init__(model, train_loader, valid_loader, Lossfunction(args),
                         optimizer, args, logger, run_id, topology, final_path)
        self.evaluator = TemporalEvaluator(args)

    def _epoch_context(self, training):
        # Training: a fresh physical ensemble per batch, fixed over its full sequence.
        if getattr(self.args, "temporal_protocol", None) == "learnable_filters":
            return nullcontext()
        return nullcontext() if training else fixed_evaluation_chips(
            self.model, temporal_configuration(self.args)["valid_seed"])

    def _resume_or_start(self):
        progress = super()._resume_or_start()
        if progress["status"] == "completed" and (
                self.final_path is None or self.store.is_completed(self.final_path)):
            self.store.restore_best(self.model, self.optimizer, self.scheduler,
                                    self.config, self.topology)
        return progress

    def _complete(self, progress):
        # Latest keeps the actual training endpoint for the existing resume protocol.
        progress = dict(progress, status="completed")
        self._save_latest(progress)
        best = self.store.restore_best(self.model, self.optimizer, self.scheduler,
                                       self.config, self.topology)
        if self.final_path is not None:
            self.store.save_completed(
                path=self.final_path, model=self.model, optimizer=self.optimizer,
                scheduler=self.scheduler,
                progress=dict(progress, selected_epoch=best["epoch"], selection="best_valid_loss"),
                config=self.config, topology=self.topology)
        return TrainingResult("completed", progress["epoch"], progress["best_valid_loss"])


GatedTrainer = TemporalTrainer  # Preserve the reviewed pLTC Python interface.


def _run_training(
    model,
    train_loader,
    valid_loader,
    lossfunction,
    optimizer,
    args,
    logger,
    run_id,
    progressive,
):
    final_path = None
    if hasattr(args, "savepath"):
        final_path = Path(os.path.join(args.savepath, f"pNN_{run_id}"))
    topology = getattr(model, "topology", [])
    result = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lossfunction=lossfunction,
        optimizer=optimizer,
        args=args,
        logger=logger,
        run_id=run_id,
        topology=topology,
        final_path=final_path,
        progressive=progressive,
    ).fit()
    return model, result.completed


def train_pnn(
    nn,
    train_loader,
    valid_loader,
    lossfunction,
    optimizer,
    args,
    logger,
    UUID="default",
):
    """Compatibility entry point for training at a fixed learning rate."""
    return _run_training(
        nn,
        train_loader,
        valid_loader,
        lossfunction,
        optimizer,
        args,
        logger,
        UUID,
        progressive=False,
    )


def train_pnn_progressive(
    nn,
    train_loader,
    valid_loader,
    lossfunction,
    optimizer,
    args,
    logger,
    UUID="default",
):
    """Compatibility entry point for training with progressive learning-rate decay."""
    return _run_training(
        nn,
        train_loader,
        valid_loader,
        lossfunction,
        optimizer,
        args,
        logger,
        UUID,
        progressive=True,
    )
