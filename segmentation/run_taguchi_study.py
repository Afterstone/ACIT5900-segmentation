import json
from pathlib import Path

import pandas as pd  # type: ignore

from segmentation.sscx import evaluate


def main(
    ta_path: Path,
):
    df_ta = pd.read_csv(ta_path)

    print(df_ta)

    CLIP_MODEL_NAME_values = [
        {'model': 'convnext_xxlarge', 'weights': 'laion2b_s34b_b82k_augreg'},
        {'model': 'convnext_large_d', 'weights': 'laion2b_s26b_b102k_augreg'},
        {'model': 'convnext_base_w', 'weights': 'laion2b_s13b_b82k_augreg'},
        {'model': 'RN50', 'weights': 'openai'},
        {'model': 'RN101', 'weights': 'openai'},
    ]
    XAI_METHOD_values = ['uniform', 'gradcam', 'gradcampp', 'layercam', ]  # TODO: Add Add two more?
    SAM_MODEL_values = ['sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth',
                        'sam_vit_h_4b8939.pth', ]  # TODO: Possibly more?
    CLIP_NORMALIZE_ATTENTION_values = ['minmax', 'standard', 'none', 'softmax', 'percentile']
    SAM_PROPOSER_N_POINTS_values = [1, 3, 5, 8, 13]
    SAM_PROPOSER_NORM_values = [1, 2, 3, 4, 5]


if __name__ == '__main__':
    main(
        ta_path=Path('taguchi_designs/L25.csv'),
    )
