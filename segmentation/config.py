from pathlib import Path

import torch as T

# MODEL_NAME: str = 'convnext_xxlarge'
# MODEL_WEIGHTS_NAME: str = 'laion2b_s34b_b82k_augreg'

CLIP_MODEL_NAME: str = 'convnext_base'
CLIP_MODEL_WEIGHTS_NAME: str = 'laion400m_s13b_b51k'

# MODEL_NAME: str = 'RN50x64'
# MODEL_WEIGHTS_NAME: str = 'openai'

# MODEL_NAME: str = 'RN101-quickgelu'
# MODEL_WEIGHTS_NAME: str = 'openai'

CLIP_CLASSIFIER_NAME: str = 'ViT-H-14'
CLIP_CLASSIFIER_WEIGHTS_NAME: str = 'laion2b_s32b_b79k'

TORCH_DEVICE: str | T.device = 'cuda' if T.cuda.is_available() else 'cpu'

# SAM_CHECKPOINT: str = "models/sam_vit_h_4b8939.pth"
# MODEL_TYPE: str = "vit_h"

SAM_CHECKPOINT: str = "models/sam_vit_b_01ec64.pth"
MODEL_TYPE: str = "vit_b"


FOODSEG103_ROOT: Path = Path('./data/FoodSeg103')
