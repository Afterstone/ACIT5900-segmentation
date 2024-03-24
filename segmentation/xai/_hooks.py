import torch as T


class GradsAndActivationsHook:
    """Attaches to a module and records its activations and gradients.

    Sources:
    - Code: Adapted from https://github.com/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb.
    """

    def __init__(self, module: T.nn.Module):
        self.data: T.Tensor = None  # type: ignore
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> T.Tensor:
        return self.data

    @property
    def gradient(self) -> T.Tensor:
        return self.data.grad   # type: ignore
