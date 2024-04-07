import typing as t

import torch as T
import torch.nn.functional as F

from segmentation.xai._basecam import BaseCam, select_layers_last
from segmentation.xai._hooks import GradsAndActivationsHook


class GradCam(BaseCam):
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
        """Computes the Grad-CAM for a given input and target.

        Sources:
        - Code: Adapted from https://github.com/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb.
        - Paper: https://arxiv.org/abs/1610.02391
        """
        if input.grad is not None:
            input.grad.data.zero_()

        requires_grad = {}
        for name, param in model.named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)

        layer = self._layers_selector(self._layers_extractor(model))[-1]
        with GradsAndActivationsHook(layer) as hook:
            output = model(input)
            output.backward(target)

            grad = hook.gradient.float()
            act = hook.activation.float()

            w = grad.mean(dim=(2, 3), keepdim=True)
            A_hat = T.sum(act * w, dim=1, keepdim=True)
            gradcam = T.clamp(A_hat, min=0)

        gradcam = F.interpolate(
            gradcam,
            input.shape[2:],
            mode='bicubic',
            align_corners=False)

        for name, param in model.named_parameters():
            param.requires_grad_(requires_grad[name])

        return gradcam
