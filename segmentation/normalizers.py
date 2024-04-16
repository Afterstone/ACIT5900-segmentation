from abc import ABC, abstractmethod
from tempfile import tempdir

import torch as T


class Normalizer(ABC):
    @abstractmethod
    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        pass


class Identity(Normalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        return tensor


class MinMaxNormalizer(Normalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        return (tensor - T.min(tensor)) / (T.max(tensor) - T.min(tensor))


class StandardNormalizer(Normalizer):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        return (tensor - T.mean(tensor)) / (T.std(tensor) + self.eps)


class SoftmaxNormalizer(Normalizer):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        return T.softmax(tensor / self.temperature, dim=-1)


class Thresholder(Normalizer):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        return T.where(tensor > self.threshold, tensor, T.zeros_like(tensor))


class PercentileThresholder(Normalizer):
    def __init__(self, percentile: float):
        super().__init__()
        self.percentile = percentile

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        return T.where(tensor > T.quantile(tensor, self.percentile), tensor, T.zeros_like(tensor))


class PolynomialStretcher(Normalizer):
    def __init__(self, degree: float):
        super().__init__()
        self.degree = degree

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        return tensor ** self.degree


class UniformIfInvalid(Normalizer):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        small_sum = tensor.sum() < self.eps
        nan_sum = T.isnan(tensor.sum()).any()
        inf_sum = T.isinf(tensor.sum()).any()
        if small_sum or nan_sum or inf_sum:
            tensor = T.ones_like(tensor)
            tensor /= tensor.sum()

        return tensor


class ToProbabilities(Normalizer):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        tensor -= tensor.min()
        tensor /= tensor.sum()
        return tensor


class SequentialNormalizer(Normalizer):
    def __init__(self, normalizers: list[Normalizer]):
        super().__init__()
        self.normalizers = normalizers

    def normalize(self, tensor: T.Tensor) -> T.Tensor:
        for normalizer in self.normalizers:
            tensor = normalizer.normalize(tensor)
        return tensor
