import torch as T
from segment_anything import SamPredictor, sam_model_registry  # type: ignore


def get_sam_model(
    sam_checkpoint: str = "models/sam_vit_h_4b8939.pth",
    sam_model_type: str = "vit_h",
    device: str | T.device = "cuda",
) -> SamPredictor:
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.eval().to(device=device)  # type: ignore
    return SamPredictor(sam)
