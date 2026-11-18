"""Sampling models for manufacturing variation."""

from abc import ABC, abstractmethod
import math

import torch


class MultiplicativeVariation(ABC):
    """Sample multiplicative manufacturing variation for nominal device parameters."""

    @abstractmethod
    def sample(self, reference, sample_count, amplitude):
        """Return multiplicative factors with shape ``[sample_count, *reference.shape]``."""


class UniformMultiplicativeVariation(MultiplicativeVariation):
    """Uniform multiplicative manufacturing variation."""

    def sample(self, reference, sample_count, amplitude):
        noise = torch.rand(
            (sample_count, *reference.shape),
            device=reference.device,
            dtype=reference.dtype,
        )
        return 1.0 + (noise * 2.0 - 1.0) * amplitude


class LogNormalMultiplicativeVariation(MultiplicativeVariation):
    """Lognormal variation with unit mean and standard deviation approximately equal to
    the supplied amplitude.
    """

    def sample(self, reference, sample_count, amplitude):
        sigma = torch.sqrt(
            torch.log1p(
                torch.as_tensor(
                    amplitude, dtype=reference.dtype, device=reference.device
                ).square()
            )
        )
        normal = torch.randn(
            (sample_count, *reference.shape),
            device=reference.device,
            dtype=reference.dtype,
        )
        return torch.exp(normal * sigma - sigma.square() / 2.0)


VARIATION_MODELS = {
    "uniform": UniformMultiplicativeVariation,
    "lognormal": LogNormalMultiplicativeVariation,
}


def create_variation(name="uniform"):
    """Create a variation sampler by registered name. Register new distributions here."""
    try:
        return VARIATION_MODELS[name]()
    except KeyError as error:
        supported = ", ".join(sorted(VARIATION_MODELS))
        raise ValueError(
            f"Unsupported variation model: {name}. Supported: {supported}."
        ) from error


class SequenceVariation(MultiplicativeVariation):
    """Cache samples for one physical module while the sequence context is active."""

    def __init__(self, base):
        self.base = base
        self.signature = None
        self.factors = None

    def sample(self, reference, sample_count, amplitude):
        if not math.isfinite(amplitude) or amplitude < 0:
            raise ValueError("Variation amplitude must be finite and nonnegative.")
        if isinstance(self.base, UniformMultiplicativeVariation) and amplitude >= 1:
            raise ValueError("Uniform variation requires amplitude < 1 for positive device values.")
        if type(sample_count) is not int or sample_count < 1:
            raise ValueError("Physical sample count must be a positive integer.")
        signature = (tuple(reference.shape), reference.dtype, reference.device, sample_count, amplitude)
        if self.signature is None:
            factors = self.base.sample(reference.detach(), sample_count, amplitude).detach()
            if not torch.isfinite(factors).all() or (factors <= 0).any():
                raise ValueError("Variation produced nonfinite or nonpositive factors.")
            self.signature, self.factors = signature, factors
        elif signature != self.signature:
            raise ValueError("Physical shape, dtype/device, samples or amplitude changed within a sequence.")
        return self.factors
