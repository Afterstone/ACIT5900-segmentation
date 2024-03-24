import torch as T
import torch.nn.functional as F

from segmentation.xai._base_cam import BaseCam
from segmentation.xai._hooks import GradsAndActivationsHook


class LayerCam(BaseCam):
    def __call__(self, model: T.nn.Module, input: T.Tensor, target: T.Tensor, layer: T.nn.Module) -> T.Tensor:
        """Computes the LayerCAM for a given input and target.

        Sources:
        - Code: Adapted from https://github.com/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb.
        - Paper: https://ieeexplore.ieee.org/document/9462463
        """
        # Zero out any gradients at the input.
        if input.grad is not None:
            input.grad.data.zero_()

        # Disable gradient settings.
        requires_grad = {}
        for name, param in model.named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)

        # Attach a hook to the model at the desired layer.
        assert isinstance(layer, T.nn.Module)
        with GradsAndActivationsHook(layer) as hook:
            # Do a forward and backward pass.
            output = model(input)
            output.backward(target)

            grad = hook.gradient.float()
            act = hook.activation.float()

            # Global average pool gradient across spatial dimension
            # to obtain importance weights.
            w = T.clamp(grad.mean(dim=(2, 3), keepdim=True), min=0)
            # Weighted combination of activation maps over channel
            # dimension.
            A_hat = T.sum(act * w, dim=1, keepdim=True)
            # We only want neurons with positive influence so we
            # clamp any negative ones.
            layer_cam = T.clamp(A_hat, min=0)

        # Resize gradcam to input resolution.
        layer_cam = F.interpolate(
            layer_cam,
            input.shape[2:],
            mode='bicubic',
            align_corners=False)

        # Restore gradient settings.
        for name, param in model.named_parameters():
            param.requires_grad_(requires_grad[name])

        return layer_cam
