import typing as t

import torch as T
import torch.nn.functional as F

from segmentation.xai._basecam import BaseCam, select_layers_last
from segmentation.xai._hooks import GradsAndActivationsHook


class GradCamPP(BaseCam):
    def __init__(
        self,
        layers_extractor: t.Callable[[T.nn.Module], t.List[T.nn.Module]],
        layers_selector: t.Callable[[list[T.nn.Module]], list[T.nn.Module]] = select_layers_last(),
        eps: float = 1e-9,
    ) -> None:
        super().__init__(
            layers_extractor=layers_extractor,
            layers_selector=layers_selector,
        )
        self.eps = eps

    def __call__(self, model: T.nn.Module, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        """Computes the Grad-CAM++ for a given input and target.

        Sources:
        - Code: Adapted from https://github.com/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb.
        - Paper: https://arxiv.org/abs/1710.11063
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

            g_sq = grad ** 2
            g_cubed = g_sq * grad
            act_mean = act.mean(dim=(2, 3), keepdim=True)

            alpha = g_sq / (2 * g_sq + act_mean * g_cubed + self.eps)

            w = (alpha * T.clamp(grad, min=0)).mean(dim=(2, 3), keepdim=True)
            A_hat = T.sum(act * w, dim=1, keepdim=True)
            gradcampp = T.clamp(A_hat, min=0)

        gradcampp = F.interpolate(
            gradcampp,
            input.shape[2:],
            mode='bicubic',
            align_corners=False
        )

        for name, param in model.named_parameters():
            param.requires_grad_(requires_grad[name])

        return gradcampp
