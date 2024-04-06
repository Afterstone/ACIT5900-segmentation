import typing as t
from abc import ABC, abstractmethod

import torch as T


def select_layers_single(index: int) -> t.Callable[[list[T.nn.Module]], list[T.nn.Module]]:
    def function(layers: list[T.nn.Module]) -> list[T.nn.Module]:
        return [layers[index]]
    return function


def select_layers_last() -> t.Callable[[list[T.nn.Module]], list[T.nn.Module]]:
    return select_layers_single(-1)


def select_layers_range(start: int, end: int | None = None) -> t.Callable[[list[T.nn.Module]], list[T.nn.Module]]:
    def function(layers: list[T.nn.Module]) -> list[T.nn.Module]:
        if end is None:
            return layers[start:]

        return layers[start:end]
    return function


class BaseCam(ABC):
    def __init__(
        self,
        layers_extractor: t.Callable[[T.nn.Module], t.List[T.nn.Module]],
        layers_selector: t.Callable[[list[T.nn.Module]], list[T.nn.Module]],
    ):
        self._layers_extractor = layers_extractor
        self._layers_selector = layers_selector

    @abstractmethod
    def __call__(self, model: T.nn.Module, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        raise NotImplementedError
