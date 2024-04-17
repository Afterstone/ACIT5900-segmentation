import json
import pickle
import time
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
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=5,
            multivariate=True,
            group=True
        )
        save_pickle(path, sampler)

    return sampler


def get_pruner(
    path: Path,
) -> optuna.pruners.BasePruner:
    pruner: optuna.pruners.BasePruner
    if path.exists():
        pruner = load_pickle(path)
    else:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=30, n_startup_trials=5)
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


def evaluate(
    clip_model_name: str,
    clip_model_weights_name: str,
    xai_method: str,
    clip_normalize_attention: str,
    sam_model_checkpoint: Path,
    sam_model_type: str,
    sam_proposer_n_points: int,
    sam_proposer_norm: float,
    dataset: FoodSegDataset,
    device: str,
    print_results_interval: int,

):
    clip_attn_mapper_config = sscx.ClipAttentionMapperConfig(
        model_name=clip_model_name,
        model_weights_name=clip_model_weights_name,
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
    )

    return res


def train(
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
        'RN50': 'openai',
        'RN101': 'openai',
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
    base_dir: Path,
    study_name: str,
    total_trials: int,
    sam_model_checkpoint: Path,
    sam_model_type: str,
):
    study_dir = base_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    db_uri = f"sqlite:///{str(study_dir)}/{study_name}.db"
    sampler = get_sampler(study_dir / Path("sampler.pkl"))
    pruner = get_pruner(study_dir / Path("pruner.pkl"))

    study = optuna.create_study(
        study_name=study_name,
        storage=db_uri,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction=optuna.study.StudyDirection.MAXIMIZE,
    )
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining_trials = total_trials - len(valid_trials)

    if remaining_trials > 0:
        print("--- Starting trials ---")
        print(f"Remaining trials: {len(valid_trials)} / {total_trials}")
        dataset = FoodSegDataset.load_pickle(config.FOODSEG103_ROOT / 'processed_train_subset')
        study.optimize(
            partial(
                train,
                sam_model_checkpoint=sam_model_checkpoint,
                sam_model_type=sam_model_type,
                dataset=dataset,
                device=config.TORCH_DEVICE,  # type: ignore
                print_results_interval=-1,
            ),  # type: ignore
            n_trials=remaining_trials,
            catch=(
                Exception,
            ),
        )
        del dataset

    print("--- Trials finished---")
    results_dict = {
        'best_trial': study.best_trial.number,
        'best_params': study.best_params,
        'train_mIoU': study.best_trial.user_attrs['mIoU'],
        'train_aAcc': study.best_trial.user_attrs['aAcc'],
    }
    print(f"Best trial: {study.best_trial.number} / {total_trials}")
    print(f"Best params:")
    for k, v in sorted(study.best_params.items(), key=lambda x: x[0]):
        v_formatted = f"{v:.4f}" if isinstance(v, float) else v
        print(f"    {k}: {v_formatted}")
    print(f"Train mIoU: {study.best_trial.user_attrs['mIoU']:.4f}")
    print(f"Train aAcc: {study.best_trial.user_attrs['aAcc']:.4f}")

    print()
    print("Evaluating...")

    clip_model_name_to_weights = {
        'convnext_xxlarge': 'laion2b_s34b_b82k_augreg',
        'convnext_base_w': 'laion2b_s13b_b82k_augreg',
    }
    test_dataset = FoodSegDataset.load_pickle(config.FOODSEG103_ROOT / 'processed_test')
    res = evaluate(
        clip_model_name=study.best_params['CLIP_MODEL_NAME'],
        clip_model_weights_name=clip_model_name_to_weights[study.best_params['CLIP_MODEL_NAME']],
        xai_method=study.best_params['XAI_METHOD'],
        clip_normalize_attention=study.best_params['CLIP_NORMALIZE_ATTENTION'],
        sam_model_checkpoint=sam_model_checkpoint,
        sam_model_type=sam_model_type,
        sam_proposer_n_points=study.best_params['SAM_PROPOSER_N_POINTS'],
        sam_proposer_norm=study.best_params['SAM_PROPOSER_NORM'],
        dataset=test_dataset,
        device=config.TORCH_DEVICE,  # type: ignore
        print_results_interval=10,
    )

    results_dict['test_mIoU'] = res.mIoU
    results_dict['test_aAcc'] = res.aAcc

    print("--- Test results ---")
    print(f"Test mIoU: {res.mIoU:.4f}")
    print(f"Test aAcc: {res.aAcc:.4f}")

    with open(study_dir / Path("results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)


if __name__ == '__main__':
    main(
        base_dir=Path("studies"),
        study_name="sscx",
        total_trials=30,
        sam_model_checkpoint=Path("models/sam_vit_b_01ec64.pth"),
        sam_model_type="vit_b",
    )
