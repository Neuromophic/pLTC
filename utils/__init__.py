import os
import random
import re
import math
import hashlib
import json
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from .training import *
from .training import serialise_configuration
from .evaluation import *
from .logger import *
from .Loader import *


def ParseBool(value):
    """Parse common command-line Boolean values."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}. Use true/false, 1/0, or yes/no.")


def FormatUUIDValue(value):
    if isinstance(value, torch.device):
        value = str(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:03d}" if value >= 0 else f"-{abs(value):03d}"
    if isinstance(value, float):
        text = f"{value:.5f}".rstrip("0").rstrip(".")
        return "0" if text in {"", "-0"} else text
    if isinstance(value, (list, tuple)):
        return "-".join(FormatUUIDValue(item) for item in value)
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "-", text).strip("._-") or "empty"


def CreateUUID(args, dataset_name):
    """Build a stable, Windows-safe run ID from the effective experiment parameters."""
    excluded = {
        "SEED",
        "projectname",
        "temppath",
        "logfilepath",
        "recordpath",
        "savepath",
    }
    if args.task == "temporal":
        configuration = {k: v for k, v in serialise_configuration(args).items() if k not in excluded}
        digest = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
        model = getattr(args, "temporal_model", None)
        family = model if model in {"prnn", "prnn2", "rnn", "gru"} else "gated"
        return f"dataset-{FormatUUIDValue(dataset_name)[:40]}_seed-{args.SEED}_{family}-{digest}"
    configuration = {k: v for k, v in serialise_configuration(args).items() if k not in excluded}
    configuration["dataset_name"] = dataset_name
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    return f"dataset-{FormatUUIDValue(dataset_name)[:40]}_seed-{args.SEED}_{digest}"


def CheckDevice(args):
    if isinstance(args.DEVICE, torch.device):
        return args.DEVICE

    if args.DEVICE == "cpu":
        return torch.device("cpu")
    elif args.DEVICE == "gpu" or args.DEVICE == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise RuntimeError("No GPU available!")
    raise ValueError(f"Unexpected device format: {args.DEVICE}")


def CodeFingerprint():
    """Fingerprint the Python implementation and circuit surrogate assets, excluding tests
    and run artifacts.
    """
    root = Path(__file__).resolve().parents[1]
    paths = [p for p in root.rglob("*.py") if "tests" not in p.relative_to(root).parts]
    paths += list((root / "models" / "assets").glob("*.package"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def SetSeed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def MakeFolder(args):
    if args.projectname:
        args.temppath = os.path.join(".", args.projectname, "temp")
        args.logfilepath = os.path.join(".", args.projectname, "log")
        args.recordpath = os.path.join(".", args.projectname, "record")
        args.savepath = os.path.join(".", args.projectname, "models")

    os.makedirs(args.temppath, exist_ok=True)

    os.makedirs(args.logfilepath, exist_ok=True)

    os.makedirs(args.savepath, exist_ok=True)

    if args.recording:
        os.makedirs(args.recordpath, exist_ok=True)


def ValidateRuntimeConfiguration(args):
    """Validate sampling and physical simulation settings before building the model."""
    positive_counts = (
        "N_train",
        "N_test",
        "R_train",
        "R_test",
        "M_train",
        "K_train",
        "M_test",
        "K_test",
        "t_test_max",
    )
    for name in positive_counts:
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")

    if (
        hasattr(args, "batch_size")
        and args.batch_size is not None
        and args.batch_size < 1
    ):
        raise ValueError(
            "batch_size must be positive or omitted for full-batch training."
        )
    if hasattr(args, "num_workers") and args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    for name in ("gmin", "gmax", "pgmin"):
        if hasattr(args, name):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
    if hasattr(args, "gmin") and hasattr(args, "gmax") and args.gmin > args.gmax:
        raise ValueError("gmin must not exceed gmax.")

    for name in ("InputNoise", "IN_test", "e_train", "e_test"):
        if hasattr(args, name):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")

    upper = getattr(args, "initial_range", None)
    if upper is not None:
        if not math.isfinite(upper) or upper <= 0:
            raise ValueError("Require a finite positive --initial-range HIGH.")
        if getattr(args, "initial_state", None) != "uniform":
            raise ValueError("--initial-range requires --initial-state uniform.")
    for name in ("gate_lr_scale", "target_lr_scale"):
        value = getattr(args, name, 1.0)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    return args


def FormulateArgs(args, resolve_device=True):
    """Fill in parameters implied by the training settings. Command generators can skip
    device detection.
    """
    ValidateRuntimeConfiguration(args)
    if args.task == "temporal":
        reference = getattr(args, "temporal_protocol", None) == "learnable_filters"
        if reference:
            if getattr(args, "temporal_encoding", None) != "stored" or args.temporal_loss != "mean":
                raise ValueError("Reference protocol requires stored inputs and mean temporal CE.")
            if args.FittingLoss != "ce" or args.learnable_group != "w":
                raise ValueError("Reference protocol requires CE and frozen nonlinear circuits (w only).")
            if args.batch_size is not None:
                raise ValueError("Reference protocol uses full-batch training; omit batch-size.")
        coupling = getattr(args, "coupling_factor", None)
        if coupling is not None and (not math.isfinite(coupling) or not 0 <= coupling <= 0.3):
            raise ValueError("coupling-factor must be finite in [0,0.3].")
        model = getattr(args, "temporal_model", None) or "pltc"
        field = "gated" if model == "pltc" else "recurrent"
        unused = "recurrent" if model == "pltc" else "gated"
        flag = "gated-config" if model == "pltc" else "recurrent-config"
        if not isinstance(getattr(args, field, None), dict):
            raise ValueError(f"Temporal training requires explicit --{flag} JSON.")
        if getattr(args, unused, None) is not None:
            raise ValueError("Provide only the configuration for the selected temporal model.")
        if args.MODE != "nominal" or args.powerestimator != "none" or args.pathnorm:
            raise ValueError("Temporal task supports nominal mode without power/pathnorm regularization.")
        if model == "pltc" and (len(args.hidden) != 1 or args.circuit_sharing not in {"layer", "neuron"}):
            raise ValueError("pLTC requires one hidden layer and layer/neuron circuit sharing.")
        if model in {"prnn", "prnn2"} and (args.circuit_sharing != "network" or args.activation_circuit != "tanh"
                                or args.e_train != 0 or args.e_test != 0):
            raise ValueError("pRNN requires network-shared tanh and zero manufacturing variation.")
        if model in {"rnn", "gru"} and (args.e_train != 0 or args.e_test != 0 or coupling not in (None, 0)):
            raise ValueError("Software baselines have no manufacturing variation or filter coupling.")
        if args.InputNoise != 0 or args.IN_test != 0:
            raise ValueError("Temporal task currently requires zero input noise.")
    elif any(getattr(args, field, None) is not None
             for field in ("gated", "recurrent", "temporal_model", "voltage_range", "temporal_loss", "temporal_encoding",
                           "temporal_protocol", "temporal_readout", "initial_state", "initial_range",
                           "coupling_factor")):
        raise ValueError("Temporal model/configuration and voltage-range require --task temporal.")
    if resolve_device:
        args.DEVICE = CheckDevice(args)
    if args.e_train == 0.0:
        args.N_train = 1
    if args.MODE == "nominal":
        args.M_train = 1
        args.K_train = 1
    if args.InputNoise == 0.0:
        args.R_train = 1
    return ValidateRuntimeConfiguration(args)


from .pruning import save_pruned_model as save_pruned_model  # noqa: E402
