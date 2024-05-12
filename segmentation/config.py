import os
from pathlib import Path

import torch as T
from dotenv import load_dotenv

load_dotenv()

# CLIP_NORMALIZE_ATTENTION: str = "minmax"
CLIP_NORMALIZE_ATTENTION: str = "standard"
# CLIP_NORMALIZE_ATTENTION: str = "softmax"
# CLIP_NORMALIZE_ATTENTION: str = "percentile"
# CLIP_NORMALIZE_ATTENTION: str = "none"
CLIP_NORMALIZER_NORM: int = 3
SAM_PROPOSER_N_POINTS: int = 3

# XAI_METHOD: str = 'gradcam'
# XAI_METHOD: str = 'gradcampp'
# XAI_METHOD: str = 'layercam'
XAI_METHOD: str = 'uniform'


# CLIP_MODEL_NAME: str = 'convnext_xxlarge'
# CLIP_MODEL_WEIGHTS_NAME: str = 'laion2b_s34b_b82k_augreg'

CLIP_MODEL_NAME: str = 'convnext_base'
CLIP_MODEL_WEIGHTS_NAME: str = 'laion400m_s13b_b51k'

# CLIP_MODEL_NAME: str = 'convnext_base_w'
# CLIP_MODEL_WEIGHTS_NAME: str = 'laion2b_s13b_b82k_augreg'

# CLIP_MODEL_NAME: str = 'RN50x64'
# CLIP_MODEL_WEIGHTS_NAME: str = 'openai'

# CLIP_MODEL_NAME: str = 'RN50'
# CLIP_MODEL_WEIGHTS_NAME: str = 'openai'

# CLIP_MODEL_NAME: str = 'RN101-quickgelu'
# CLIP_MODEL_WEIGHTS_NAME: str = 'openai'

# CLIP_CLASSIFIER_NAME: str = 'ViT-H-14'
# CLIP_CLASSIFIER_WEIGHTS_NAME: str = 'laion2b_s32b_b79k'

CLIP_CLASSIFIER_NAME: str = 'ViT-B-16'
CLIP_CLASSIFIER_WEIGHTS_NAME: str = 'laion2b_s34b_b88k'

TORCH_DEVICE: str | T.device = 'cuda' if T.cuda.is_available() else 'cpu'

# SAM_CHECKPOINT: str = "models/sam_vit_h_4b8939.pth"
# MODEL_TYPE: str = "vit_h"

# SAM_CHECKPOINT: str = "models/sam_vit_l_0b3195.pth"
# MODEL_TYPE: str = "vit_l"

SAM_CHECKPOINT: str = "models/sam_vit_b_01ec64.pth"
MODEL_TYPE: str = "vit_b"


FOODSEG103_ROOT: Path = Path('./data/FoodSeg103')
UECFOODPIXCOMPLETE_ROOT: Path = Path('./data/UECFoodPixComplete')

PRINT_RESULTS_EVERY_N_STEPS: int = 10

FOOD50SEG_PASSWORD: str | None = os.getenv("FOOD50SEG_PASSWORD")
