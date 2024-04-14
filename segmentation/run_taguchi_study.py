import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tqdm
from regex import P  # type: ignore

import segmentation.config as config
import segmentation.sscx as sscx
from segmentation.datasets.foodseg103 import FoodSegDataset


class TaguchiParameter:
    def __init__(
        self,
        name: str,
        index: int,
        values: list,
    ):
        self.name = name
        self.index = index
        self.values = values


class TaguchiArrayBuilder:
    def __init__(
        self,
        ta_path: Path,
    ):
        self.df_ta = pd.read_csv(ta_path)
        self.n_params = len(self.df_ta.columns)
        self.params: list[TaguchiParameter] = [None] * self.n_params  # type: ignore

    def __len__(self):
        return len(self.df_ta.index)

    def add_param(self, ta_param: TaguchiParameter):
        if self.params[ta_param.index] is not None:
            raise ValueError(f'Parameter at index {ta_param.index} already exists')

        self.params[ta_param.index] = ta_param

    def ensure_initialized(self):
        if None in self.params:
            raise ValueError('Not all parameters have been initialized')

    def get_trial(self, index: int) -> dict:
        self.ensure_initialized()

        row = self.df_ta.iloc[index]
        params = {}
        for param in self.params:
            value_loc = row.iloc[param.index] - 1
            param_value = param.values[value_loc]
            params[param.name] = param_value
        return params


def main(
    ta_path: Path,
    results_dir: Path,
    model_dir: Path,
):
    # Strata parameters
    SAM_MODELS = ['sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth', 'sam_vit_h_4b8939.pth']
    SAM_MODEL_TO_TYPE = {
        'sam_vit_b_01ec64.pth': 'vit_b',
        'sam_vit_l_0b3195.pth': 'vit_l',
        'sam_vit_h_4b8939.pth': 'vit_h',
    }

    studies: dict[str, list[dict]] = defaultdict(list)
    for SAM_MODEL in SAM_MODELS:
        json_path = results_dir / Path(f'{SAM_MODEL}.json')

        if json_path.exists():
            with open(json_path, 'r') as f:
                results = json.load(f)
        else:
            study = TaguchiArrayBuilder(ta_path)
            # Taguchi parameters
            study.add_param(TaguchiParameter('CLIP_MODEL_PARAMS', 0, [
                {'model': 'convnext_xxlarge', 'weights': 'laion2b_s34b_b82k_augreg'},
                {'model': 'convnext_base_w', 'weights': 'laion2b_s13b_b82k_augreg'},
                {'model': 'RN50', 'weights': 'openai'},
                {'model': 'RN101', 'weights': 'openai'},
            ]))
            study.add_param(TaguchiParameter('XAI_METHOD', 1, ['uniform', 'gradcam', 'gradcampp', 'layercam']))
            study.add_param(TaguchiParameter('CLIP_NORMALIZE_ATTENTION', 2,
                            ['standard', 'none', 'softmax', 'percentile']))
            study.add_param(TaguchiParameter('SAM_PROPOSER_N_POINTS', 3, [1, 3, 9, 16]))
            study.add_param(TaguchiParameter('SAM_PROPOSER_NORM', 4, [1, 2, 3, 4]))

            trials = [
                {
                    'params': study.get_trial(i),
                    'mIoU': [],
                    'aAcc': [],
                }
                for i in range(len(study))
            ]

            with open(json_path, 'w') as f:
                json.dump(trials, f, indent=2, sort_keys=True)

        studies[SAM_MODEL] = results

    for SAM_MODEL, trials in studies.items():
        json_path = results_dir / Path(f'{SAM_MODEL}.json')

        print(f'Running {SAM_MODEL}')
        print(f'Number of trials: {len(trials)}')

        # Check the max number of results per trial.
        max_results = max(len(trial['mIoU']) for trial in trials)
        min_results = min(len(trial['mIoU']) for trial in trials)
        if max_results == min_results:
            max_results += 1

        for trial in tqdm.tqdm(trials, total=len(trials), desc='Running trials', unit='trial'):
            if len(trial['mIoU']) >= max_results:
                print(f"Skipping trial because it has already been run: {trial['params']}")
                continue

            print(f'Running trial: {trial["params"]}')
            trial_params = trial['params']

            res = sscx.evaluate(
                clip_attn_mapper_config=sscx.ClipAttentionMapperConfig(
                    model_name=trial_params['CLIP_MODEL_PARAMS']['model'],  # type: ignore
                    model_weights_name=trial_params['CLIP_MODEL_PARAMS']['weights'],  # type: ignore
                    xai_method=trial_params['XAI_METHOD'],  # type: ignore
                    normalize_attention=trial_params['CLIP_NORMALIZE_ATTENTION'],  # type: ignore
                    normalizer_norm=trial_params['SAM_PROPOSER_NORM'],  # type: ignore
                ),
                clip_cls_config=sscx.ClipClassifierConfig(
                    model_name=config.CLIP_CLASSIFIER_NAME,
                    model_weights_name=config.CLIP_CLASSIFIER_WEIGHTS_NAME,
                ),
                sam_config=sscx.SamConfig(
                    checkpoint=str(model_dir / SAM_MODEL),
                    model_type=SAM_MODEL_TO_TYPE[SAM_MODEL],
                    proposer_n_points=trial_params['SAM_PROPOSER_N_POINTS'],  # type: ignore
                ),
                device=config.TORCH_DEVICE,
                dataset=FoodSegDataset.load_pickle(config.FOODSEG103_ROOT / 'processed_test'),
                print_results_interval=config.PRINT_RESULTS_EVERY_N_STEPS,
            )

            trial['mIoU'].append(res.mIoU)  # type: ignore
            trial['aAcc'].append(res.aAcc)  # type: ignore

            with open(json_path, 'w') as f:
                json.dump(trials, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main(
        ta_path=Path('taguchi_designs/L16B.csv'),
        results_dir=Path('results'),
        model_dir=Path('models'),
    )
