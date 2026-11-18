"""Shared format conventions for pNN tensor datasets."""

import torch


DATASET_FORMAT_VERSION = 1
CLASSIFICATION_TASK = "classification"
REQUIRED_SPLITS = ("train", "valid", "test")


def validate_temporal_dataset(data, path):
    """Validate a .tsds dataset with saved splits and [sample, channel, time] tensors."""
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"]:
        raise ValueError(f"{path} must contain a named temporal dataset dictionary.")
    for key in ("n_feature", "n_class", "n_length"):
        if type(data.get(key)) is not int or data[key] < (2 if key == "n_class" else 1):
            raise ValueError(f"Invalid temporal dataset {key} in {path}.")
    for mode in REQUIRED_SPLITS:
        x, y = data.get(f"X_{mode}"), data.get(f"y_{mode}")
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError(f"Missing temporal tensors for {mode} in {path}.")
        if x.ndim != 3 or x.shape[1:] != (data["n_feature"], data["n_length"]) or len(x) < 1:
            raise ValueError(f"Invalid temporal shape for {mode} in {path}.")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError(f"Temporal input must be finite floating point: {mode}.")
        if y.dtype != torch.long or y.shape != (len(x),) or y.min() < 0 or y.max() >= data["n_class"]:
            raise ValueError(f"Invalid temporal labels for {mode} in {path}.")
        if f"n_{mode}" in data and data[f"n_{mode}"] != len(x):
            raise ValueError(f"Temporal sample count mismatch for {mode} in {path}.")


def validate_classification_dataset(data, path):
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a dictionary.")
    if data.get("format_version") != DATASET_FORMAT_VERSION:
        raise ValueError(
            f"{path} has format_version={data.get('format_version')!r}; "
            f"expected {DATASET_FORMAT_VERSION}."
        )
    if data.get("task") != CLASSIFICATION_TASK:
        raise ValueError(f"{path} is not a classification dataset.")
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise ValueError(f"{path} must define a non-empty dataset name.")
    if data["name"] != data["name"].lower():
        raise ValueError(f"{path} dataset name must be lowercase.")

    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"{path} must contain a splits dictionary.")
    n_feature, n_class = data.get("n_feature"), data.get("n_class")
    if not isinstance(n_feature, int) or n_feature < 1:
        raise ValueError(f"{path} has invalid n_feature={n_feature!r}.")
    if not isinstance(n_class, int) or n_class < 2:
        raise ValueError(f"{path} has invalid n_class={n_class!r}.")

    for split_name in REQUIRED_SPLITS:
        split = splits.get(split_name)
        if not isinstance(split, dict):
            raise ValueError(f"{path} is missing split {split_name!r}.")
        x, y = split.get("x"), split.get("y")
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError(f"{path}:{split_name} must contain tensor x and y.")
        if x.ndim != 2 or x.shape[1] != n_feature or len(x) == 0:
            raise ValueError(
                f"{path}:{split_name} has invalid feature shape {tuple(x.shape)}."
            )
        if not torch.is_floating_point(x) or not torch.isfinite(x).all():
            raise ValueError(
                f"{path}:{split_name} features must be finite floating-point values."
            )
        if y.ndim != 1 or len(x) != len(y):
            raise ValueError(
                f"{path}:{split_name} has incompatible label shape {tuple(y.shape)}."
            )
        if y.dtype != torch.long:
            raise ValueError(f"{path}:{split_name} labels must use torch.long.")
        if y.min().item() < 0 or y.max().item() >= n_class:
            raise ValueError(f"{path}:{split_name} labels are outside [0, {n_class}).")
