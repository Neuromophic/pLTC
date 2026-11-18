import json
import hashlib
import io
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.schema import validate_classification_dataset, validate_temporal_dataset

LEGACY_NORMAL_DATASETS = (
    "Dataset_acuteinflammation.ds",
    "Dataset_balancescale.ds",
    "Dataset_breastcancerwisc.ds",
    "Dataset_cardiotocography3clases.ds",
    "Dataset_energyy1.ds",
    "Dataset_energyy2.ds",
    "Dataset_iris.ds",
    "Dataset_mammographic.ds",
    "Dataset_pendigits.ds",
    "Dataset_seeds.ds",
    "Dataset_tictactoe.ds",
    "Dataset_vertebralcolumn2clases.ds",
    "Dataset_vertebralcolumn3clases.ds",
)

_MODE_SEED_OFFSETS = {"train": 0, "valid": 1, "test": 2}
_SEED_MODULUS = 2**63 - 1


def _dataset_path(dataset_name, data_path):
    candidate = Path(dataset_name)
    if candidate.suffix == ".ds":
        candidates = (candidate.name,)
    else:
        candidates = (f"{candidate.name}.ds", f"Dataset_{candidate.name}.ds")
    for name in candidates:
        path = Path(os.path.join(os.fspath(data_path), name))
        if path.is_file():
            return path
    raise FileNotFoundError(f"Dataset not found in {data_path}: {dataset_name}.")


class SerializedDataset(Dataset):
    """CPU-resident dataset backed by the versioned .ds tensor format."""

    def __init__(self, dataset_name, args, datapath=None, mode="train"):
        if mode not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        data_path = getattr(args, "DataPath") if datapath is None else datapath
        self.path = _dataset_path(dataset_name, data_path)
        data = torch.load(self.path, map_location="cpu", weights_only=True)
        validate_classification_dataset(data, self.path)

        split = data["splits"][mode]
        self.x = split["x"].contiguous()
        self.y = split["y"].contiguous()
        self.mode = mode
        self.seed = int(getattr(args, "SEED", 0))
        self.epoch = 0
        self.repeats = (
            getattr(args, "R_test", 1)
            if mode == "test"
            else getattr(args, "R_train", 1)
        )
        self.repeats = max(1, int(self.repeats))
        self.input_noise = (
            getattr(args, "IN_test", 0.0)
            if mode == "test"
            else getattr(args, "InputNoise", 0.0)
        )

        self.data_name = data["name"]
        self.N_class = data["n_class"]
        self.N_feature = data["n_feature"]
        self.N_train = len(data["splits"]["train"]["x"])
        self.N_valid = len(data["splits"]["valid"]["x"])
        self.N_test = len(data["splits"]["test"]["x"])
        self.metadata = data.get("metadata", {})

    def set_epoch(self, epoch):
        """Set the epoch so training noise varies reproducibly between epochs."""
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        self.epoch = epoch

    def _noise_generator(self, source_index, repeat_index):
        """Create a reproducible random stream for each physical repeat of a sample."""
        epoch = self.epoch if self.mode == "train" else 0
        seed = (
            self.seed
            + _MODE_SEED_OFFSETS[self.mode] * 1_000_003
            + epoch * 1_000_033
            + source_index * 1_000_037
            + repeat_index * 1_000_081
        ) % _SEED_MODULUS
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def __getitem__(self, index):
        source_index = index % len(self.x)
        repeat_index = index // len(self.x)
        x = self.x[source_index]
        if self.input_noise:
            noise = torch.randn(
                x.shape,
                dtype=x.dtype,
                device=x.device,
                generator=self._noise_generator(source_index, repeat_index),
            )
            x = x * (noise * self.input_noise + 1.0)
        return x, self.y[source_index]

    def __len__(self):
        return len(self.x) * self.repeats


class TemporalDataset(SerializedDataset):
    """Load stored voltages or fit the voltage encoding on the training split.

    The checkpoint records the dataset fingerprint and encoding for subsequent runs.
    """

    def __init__(self, args, datapath, mode):
        if mode not in _MODE_SEED_OFFSETS:
            raise ValueError("Unsupported temporal split.")
        name = getattr(args, "dataset_name", None)
        if not name or Path(name).name != name or Path(name).suffix != ".tsds":
            raise ValueError("Temporal task requires an explicit --dataset-name filename.tsds.")
        bounds = getattr(args, "voltage_range", None)
        if bounds is None or len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("Require a finite increasing --voltage-range LOW HIGH.")
        encoding = getattr(args, "temporal_encoding", None) or "train_minmax"
        if encoding not in {"stored", "train_minmax"}:
            raise ValueError("Temporal encoding must be stored or train_minmax.")
        self.path = Path(datapath, name).resolve()
        raw = self.path.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        validate_temporal_dataset(data, self.path)
        train = data["X_train"].double()
        lower, upper = train.amin((0, 2)), train.amax((0, 2))
        provenance = dict(version=1, path=str(self.path), sha256=fingerprint,
                          name=data["name"], n_feature=data["n_feature"], n_class=data["n_class"],
                          n_length=data["n_length"],
                          encoding="stored" if encoding == "stored" else "train_channel_minmax_clip",
                          lower=lower.tolist(), upper=upper.tolist(), voltage_range=list(bounds))
        saved = getattr(args, "temporal_data", None)
        if saved is not None and saved != provenance:
            raise ValueError("Temporal dataset or voltage encoding differs from the checkpoint.")
        if saved is None:
            if mode != "train":
                raise ValueError("Register the temporal data on the train split before validation/test.")
            args.temporal_data = provenance
        if encoding == "stored":
            # Match LearnableFilters/utils/Loader.py: retain saved values, including
            # validation/test values outside the nominal [-1, 1] reference range.
            self.x = data[f"X_{mode}"].transpose(1, 2).contiguous()
            clipped = 0.0
        else:
            x = data[f"X_{mode}"].double().transpose(1, 2)
            span = upper - lower
            scaled = (x - lower) / torch.where(span > 0, span, torch.ones_like(span))
            scaled = torch.where(span > 0, scaled, torch.full_like(scaled, 0.5))
            clipped = ((scaled < 0) | (scaled > 1)).double().mean().item()
            self.x = (bounds[0] + scaled.clamp(0, 1) * (bounds[1] - bounds[0])).float().contiguous()
        self.y = data[f"y_{mode}"].contiguous()
        self.mode, self.seed, self.epoch = mode, int(args.SEED), 0
        self.repeats = args.R_test if mode == "test" else args.R_train
        self.input_noise = args.IN_test if mode == "test" else args.InputNoise
        if self.input_noise != 0:
            raise ValueError("Temporal CLI currently requires zero input noise; voltage-domain noise is not calibrated.")
        self.data_name, self.N_feature, self.N_class = data["name"], data["n_feature"], data["n_class"]
        self.N_train, self.N_valid, self.N_test = [len(data[f"y_{s}"]) for s in _MODE_SEED_OFFSETS]
        self.metadata = dict(provenance=provenance, clipped_fraction=clipped)
        if encoding == "stored":
            self.metadata.update(
                input_min=self.x.amin((0, 1)).tolist(),
                input_max=self.x.amax((0, 1)).tolist(),
                outside_reference_fraction=((self.x < bounds[0]) | (self.x > bounds[1])).double().mean().item(),
            )


def _make_loader(dataset, mode, args):
    batch_size = getattr(args, "batch_size", None) or len(dataset)
    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive or omitted for full-batch training."
        )
    if batch_size > len(dataset):
        print(
            f"batch_size={batch_size} exceeds dataset size {len(dataset)}; using the dataset size instead."
        )
        batch_size = len(dataset)

    generator = torch.Generator()
    generator.manual_seed(getattr(args, "SEED", 0) + _MODE_SEED_OFFSETS[mode])
    device = getattr(args, "DEVICE", "cpu")
    pin_memory = getattr(device, "type", device) == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=mode == "train" and getattr(args, "temporal_protocol", None) != "learnable_filters",
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def _seed_worker(worker_id):
    """Seed Python, NumPy, and PyTorch consistently in each data loader worker."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _dataset_info(dataset):
    return {
        "dataname": dataset.data_name,
        "N_feature": dataset.N_feature,
        "N_class": dataset.N_class,
        "N_train": dataset.N_train,
        "N_valid": dataset.N_valid,
        "N_test": dataset.N_test,
        "metadata": dataset.metadata,
    }


def _normal_dataset_name(args, data_path):
    dataset = getattr(args, "dataset", None)
    if dataset:
        registry_path = Path(os.path.join(os.fspath(data_path), "registry.json"))
        if not registry_path.is_file():
            raise FileNotFoundError(
                f"Dataset registry does not exist: {registry_path}. "
                "Run dataset/refresh_dataset_registry.py first."
            )
        with registry_path.open(encoding="utf-8") as stream:
            registry = json.load(stream)
        try:
            return registry["datasets"][dataset]["file"]
        except KeyError as error:
            raise ValueError(f"Unknown dataset name: {dataset}.") from error

    explicit_name = getattr(args, "dataset_name", None)
    if explicit_name:
        return explicit_name

    index = getattr(args, "DATASET", 0)
    if not 0 <= index < len(LEGACY_NORMAL_DATASETS):
        raise ValueError(f"DATASET must be in [0, {len(LEGACY_NORMAL_DATASETS) - 1}].")
    return LEGACY_NORMAL_DATASETS[index]


def GetDataLoader(args, mode, path=None):
    """Build a CPU DataLoader. The caller moves each batch to the target device."""
    data_path = os.fspath(getattr(args, "DataPath") if path is None else path)

    if args.task == "temporal":
        args.DataPath = str(Path(data_path).resolve())
        selected = TemporalDataset(args, Path(data_path, "temporal"), mode)
        return _make_loader(selected, mode, args), _dataset_info(selected)

    if args.task == "normal":
        normal_path = os.path.join(data_path, "normal")
        dataset_name = _normal_dataset_name(args, normal_path)
        selected = SerializedDataset(dataset_name, args, normal_path, mode)
        return _make_loader(selected, mode, args), _dataset_info(selected)

    if args.task == "split":
        raise NotImplementedError(
            "Split datasets are stored in dataset/split, but superpNN training is not implemented yet."
        )

    raise ValueError(f"Unsupported task: {args.task}")


# Keep the lowercase class name for compatibility with older callers.
dataset = SerializedDataset
