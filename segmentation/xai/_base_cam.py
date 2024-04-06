from typing import Protocol

import torch as T


class BaseCam(Protocol):
    def __call__(self, model: T.nn.Module, input: T.Tensor, target: T.Tensor, layer: T.nn.Module | list[T.nn.Module]) -> T.Tensor:
        ...
