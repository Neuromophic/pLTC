"""Classification loss and power regularization for pNN models."""

import torch

from .regularization import default_weight_regularization


class Lossfunction(torch.nn.Module):
    """Combine classification loss with an optional physical power penalty."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight_regularization = default_weight_regularization()

    def standard(self, prediction, label):
        label = label.reshape(-1, 1)
        correct = prediction.gather(1, label)
        others = prediction.scatter(1, label, -(10**10)).max(dim=1, keepdim=True).values
        zero = torch.zeros_like(correct)
        return torch.mean(
            torch.maximum(self.args.m + self.args.T - correct, zero)
            + torch.maximum(self.args.m + others, zero)
        )

    def forward(self, nn, output, label, power):
        if self.args.FittingLoss == "ce":
            fitting_loss = torch.stack(
                [torch.nn.functional.cross_entropy(sample, label) for sample in output]
            ).mean()
        elif self.args.FittingLoss == "standard":
            fitting_loss = torch.stack(
                [self.standard(sample, label) for sample in output]
            ).mean()
        else:
            raise ValueError(f"Unsupported FittingLoss: {self.args.FittingLoss}.")
        if self.args.powerestimator == "none":
            return fitting_loss
        if self.args.powerestimator == "attraction":
            penalty = self.weight_regularization(nn)
        elif self.args.powerestimator == "power":
            penalty = power
        elif self.args.powerestimator == "both":
            penalty = (
                self.args.estimatorbalance * power
                + (1.0 - self.args.estimatorbalance) * self.weight_regularization(nn)
            ) / 2.0
        else:
            raise ValueError(f"Unsupported powerestimator: {self.args.powerestimator}.")
        return (
            1.0 - self.args.powerbalance
        ) * fitting_loss + self.args.powerbalance * penalty
