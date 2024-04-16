import typing as t

import torch as T

from segmentation.xai._basecam import BaseCam, select_layers_last


class UniformXai(BaseCam):
    def __init__(
        self,
        layers_extractor: t.Callable[[T.nn.Module], t.List[T.nn.Module]],
        layers_selector: t.Callable[[list[T.nn.Module]], list[T.nn.Module]] = select_layers_last(),
    ):
        super().__init__(
            layers_extractor=layers_extractor,
            layers_selector=layers_selector,
        )

    def __call__(self, model: T.nn.Module, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        """Baseline: Returns uniform probability over all pixels.
        """

        return T.ones_like(input[0, 0, :, :])
