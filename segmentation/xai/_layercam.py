import typing as t
from contextlib import ExitStack

import torch as T
import torch.nn.functional as F

from segmentation.xai._basecam import BaseCam, select_layers_range
from segmentation.xai._hooks import GradsAndActivationsHook


class LayerCam(BaseCam):
    def __init__(
        self,
        layers_extractor: t.Callable[[T.nn.Module], t.List[T.nn.Module]],
        layers_selector: t.Callable[[list[T.nn.Module]], list[T.nn.Module]] = select_layers_range(start=-2),
    ):
        super().__init__(
            layers_extractor=layers_extractor,
            layers_selector=layers_selector,
        )

    def __call__(self, model: T.nn.Module, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        """Computes the LayerCAM for a given input and target.

        Sources:
        - Code: Adapted from https://github.com/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb.
        - Paper: https://ieeexplore.ieee.org/document/9462463
        """

        if input.grad is not None:
            input.grad.data.zero_()

        requires_grad = {}
        for name, param in model.named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)

        layers = self._layers_selector(self._layers_extractor(model))
        with ExitStack() as stack:
            hooks = [stack.enter_context(GradsAndActivationsHook(layer)) for layer in layers]

            output = model(input)
            output.backward(target)

            cams: list[T.Tensor] = []
            for hook in hooks:
                grad = hook.gradient.float()
                act = hook.activation.float()

                w = T.clamp(grad.mean(dim=(2, 3), keepdim=True), min=0)
                A_hat = T.sum(act * w, dim=1, keepdim=True)
                layer_cam = T.clamp(A_hat, min=0)
                cams.append(layer_cam)

            resized_cams: list[T.Tensor] = []
            for cam in cams:
                resized_cams.append(F.interpolate(
                    cam,
                    input.shape[2:],
                    mode='bicubic',
                    align_corners=False,
                ))

            layer_cam = T.stack(resized_cams).mean(dim=0, keepdim=True).squeeze(0)

        for name, param in model.named_parameters():
            param.requires_grad_(requires_grad[name])

        return layer_cam
