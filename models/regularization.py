"""Composable network regularization terms."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class NetworkRegularization(nn.Module, ABC):
    """Base class for regularization terms. Implement ``forward`` to add a term."""

    @abstractmethod
    def forward(self, network):
        """Return a scalar penalty for the current network parameters."""


class WeightAttractionRegularization(NetworkRegularization):
    """Encourage conductances in each output column to cluster."""

    def forward(self, network):
        values = []
        for layer in network.layers:
            theta = layer.crossbar.theta
            values.append((theta - theta.mean(dim=0, keepdim=True)).square().mean())
        return torch.stack(values).mean()


class L1ConductanceRegularization(NetworkRegularization):
    """Encourage sparse crossbar conductances with an L1 penalty."""

    def forward(self, network):
        return torch.stack(
            [layer.crossbar.theta.abs().mean() for layer in network.layers]
        ).mean()


class MeanRegularization(NetworkRegularization):
    """Average the supplied regularization terms."""

    def __init__(self, terms):
        super().__init__()
        if not terms:
            raise ValueError("terms must not be empty.")
        self.terms = nn.ModuleList(terms)

    def forward(self, network):
        return torch.stack([term(network) for term in self.terms]).mean()


def default_weight_regularization():
    """Return the default combination used by the original attraction strategy."""
    return MeanRegularization(
        [
            WeightAttractionRegularization(),
            L1ConductanceRegularization(),
        ]
    )
