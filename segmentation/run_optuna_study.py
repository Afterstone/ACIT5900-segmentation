import pickle
import time
import typing as t
from functools import partial
from pathlib import Path

import optuna

import segmentation.config as config
import segmentation.sscx as sscx
from segmentation.datasets.foodseg103 import FoodSegDataset


def save_pickle(path: Path, obj: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_sampler(
    path: Path,
) -> optuna.samplers.BaseSampler:
    sampler: optuna.samplers.BaseSampler
    if path.exists():
        sampler = load_pickle(path)
    else:
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
        save_pickle(path, sampler)

    return sampler


def get_pruner(
    path: Path,
) -> optuna.pruners.BasePruner:
    pruner: optuna.pruners.BasePruner
    if path.exists():
        pruner = load_pickle(path)
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=100)
        save_pickle(path, pruner)

    return pruner


class ProgressCallback:
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def __call__(self, step: int, results: dict[str, str | float | int]):
        miou: float = results['miou']  # type: ignore
        self.trial.report(miou, step)

        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(
    trial: optuna.Trial,
    sam_model_checkpoint: Path,
    sam_model_type: str,
    dataset: FoodSegDataset,
    device: str,
    print_results_interval: int,
) -> float:
    start_time = time.monotonic_ns()

    clip_model_name_to_weights = {
        'convnext_xxlarge': 'laion2b_s34b_b82k_augreg',
        'convnext_base_w': 'laion2b_s13b_b82k_augreg',
    }

    clip_model_names: list[str] = sorted(list(clip_model_name_to_weights.keys()))
    clip_model_name: str = trial.suggest_categorical('CLIP_MODEL_NAME', clip_model_names)
    xai_method: str = trial.suggest_categorical('XAI_METHOD', ['uniform', 'gradcam', 'gradcampp', 'layercam'])
    clip_normalize_attention: str = trial.suggest_categorical(
        'CLIP_NORMALIZE_ATTENTION',
        ['standard', 'none', 'softmax', 'percentile']
    )
    sam_proposer_n_points: int = trial.suggest_int('SAM_PROPOSER_N_POINTS', 1, 32)
    sam_proposer_norm: float = trial.suggest_float('SAM_PROPOSER_NORM', 1.0, 5.0)

    clip_attn_mapper_config = sscx.ClipAttentionMapperConfig(
        model_name=clip_model_name,
        model_weights_name=clip_model_name_to_weights[clip_model_name],
        xai_method=xai_method,
        normalize_attention=clip_normalize_attention,
        normalizer_norm=sam_proposer_norm,
    )
    clip_cls_config = sscx.ClipClassifierConfig(
        model_name=config.CLIP_CLASSIFIER_NAME,
        model_weights_name=config.CLIP_CLASSIFIER_WEIGHTS_NAME,
    )
    sam_config = sscx.SamConfig(
        checkpoint=str(sam_model_checkpoint),
        model_type=sam_model_type,
        proposer_n_points=sam_proposer_n_points,
    )

    res = sscx.evaluate(
        clip_attn_mapper_config=clip_attn_mapper_config,
        clip_cls_config=clip_cls_config,
        sam_config=sam_config,
        device=device,
        dataset=dataset,
        print_results_interval=print_results_interval,
        progress_callback=ProgressCallback(trial),
    )

    trial.set_user_attr('mIoU', res.mIoU)
    trial.set_user_attr('aAcc', res.aAcc)

    total_time_sec = (time.monotonic_ns() - start_time) / 1e9
    total_time_min = total_time_sec / 60
    trial.set_user_attr('elapsed_minutes', total_time_min)

    return res.mIoU


def main(
    base_dir: Path = Path("studies"),
    study_name: str = "sscx",
    total_trials: int = 100,
):
    study_dir = base_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    db_uri = f"sqlite:///{str(study_dir)}/{study_name}.db"
    sampler = get_sampler(study_dir / Path("sampler.pkl"))
    pruner = get_pruner(study_dir / Path("pruner.pkl"))

    dataset = FoodSegDataset.load_pickle(config.FOODSEG103_ROOT / 'processed_test')

    study = optuna.create_study(
        study_name=study_name,
        storage=db_uri,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction=optuna.study.StudyDirection.MAXIMIZE,
    )
    remaining_trials = total_trials - len(study.trials)

    if remaining_trials > 0:
        study.optimize(
            partial(
                objective,
                sam_model_checkpoint=Path("models/sam_vit_b_01ec64.pth"),
                sam_model_type="vit_b",
                dataset=dataset,
                device=config.TORCH_DEVICE,  # type: ignore
                print_results_interval=-1,

            ),  # type: ignore
            n_trials=remaining_trials
        )


if __name__ == '__main__':
    main()
