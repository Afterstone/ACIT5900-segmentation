import datetime as dt
import typing as t
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import PIL.Image as Image
import torch as T
import torch.nn.functional as F
import tqdm
from sklearn.model_selection import KFold  # type: ignore
from sympy import Q
from torch.utils.data import (DataLoader, Dataset, Subset,  # noqa: E402
                              TensorDataset, random_split)

from segmentation.datasets.foodseg103 import FoodSegDataset
from segmentation.models import get_clip_model


def get_embeddings(
    data_folder: Path,
    model_name: str,
    model_weights: str,
) -> tuple[T.Tensor, list[list[int]]]:
    image_classes: list[list[int]]

    saved_embeddings = data_folder / f'embeddings_{model_name}_{model_weights}.pt'

    if saved_embeddings.exists():
        embeddings = T.load(saved_embeddings)
        X = embeddings['X']
        image_classes = embeddings['image_classes']
    else:
        model, preprocess, _ = get_clip_model(
            model_name=model_name,
            model_weights_name=model_weights,
            device=T.device('cuda')
        )

        dataset = FoodSegDataset.load_pickle(data_folder)

        image_embeddings: list[T.Tensor] = []
        image_classes = []
        for idx in tqdm.trange(len(dataset), desc=f"Generating embeddings for {model_name} / {model_weights}", leave=False):
            img_path = dataset.image_paths[idx]
            with Image.open(img_path) as img, T.no_grad(), T.cuda.amp.autocast():  # type: ignore
                X = preprocess(img).unsqueeze(0).to(T.device('cuda'))  # type: ignore

                image_embedding: T.Tensor = model.encode_image(X).detach().cpu().squeeze().to(T.float32)  # type: ignore
                image_embeddings.append(image_embedding)

            y = dataset.y[idx].unique().tolist()
            image_classes.append(y)

        X = T.stack(image_embeddings)

        T.save({'X': X, 'image_classes': image_classes}, saved_embeddings)

    return X, image_classes


@dataclass
class ClipModelLoadConfig:
    model_name: str
    model_weights: str


@dataclass
class EmbeddingLoadConfig:
    model_name: str
    model_weights: str
    type_: str
    path: Path


@dataclass
class Embedding:
    model_name: str
    model_weights: str
    X: T.Tensor
    image_classes: list[list[int]]
    type_: str

    def get_data(self, n_classes: int | None = None) -> tuple[T.Tensor, T.Tensor]:
        if n_classes is None:
            universe = set()
            for classes in self.image_classes:
                universe.update(classes)
            n_classes = len(universe)

        X = self.X
        y = T.zeros((len(self.image_classes), n_classes), dtype=T.long)

        for i, classes in enumerate(self.image_classes):
            for c in classes:
                y[i, c] = 1

        return X, y


class EmbeddingRegistry:
    def __init__(
        self,
        types: t.Iterable[str] = frozenset({'train', 'test'}),
    ) -> None:
        self.registered_types = set(types)
        self.embeddings: dict[str, Embedding] = {}

    def _validate_type(self, type_: str) -> None:
        if type_ not in self.registered_types:
            raise ValueError(f"Unknown type: {type_}")

    def add_type(self, type_: str) -> None:
        self.registered_types.add(type_)

    def construct_embedding_key(self, model_name: str, model_weights: str, type_: str) -> str:
        self._validate_type(type_)
        return f"{model_name=} / {model_weights=} / {type_=}"

    def get(
        self,
        model_name: str,
        model_weights: str,
        type_: str,
    ) -> Embedding:
        key = self.construct_embedding_key(model_name, model_weights, type_)
        if key not in self.embeddings:
            raise ValueError(f"Embedding not found: {key}")

        return self.embeddings[key]

    def search(
        self,
        model_name: str | None,
        model_weights: str | None,
        type_: str | None,
    ) -> list[Embedding]:
        embeddings = []

        for embedding in self.embeddings.values():
            if model_name is not None and embedding.model_name != model_name:
                continue
            if model_weights is not None and embedding.model_weights != model_weights:
                continue
            if type_ is not None and embedding.type_ != type_:
                continue

            embeddings.append(embedding)

        return embeddings

    @staticmethod
    def load(config: EmbeddingLoadConfig) -> Embedding:
        X, image_classes = get_embeddings(
            data_folder=config.path,
            model_name=config.model_name,
            model_weights=config.model_weights,
        )
        embedding = Embedding(
            model_name=config.model_name,
            model_weights=config.model_weights,
            X=X,
            image_classes=image_classes,
            type_=config.type_,
        )
        return embedding

    def register(self, embedding: Embedding) -> None:
        self._validate_type(embedding.type_)
        key = self.construct_embedding_key(embedding.model_name, embedding.model_weights, embedding.type_)
        self.embeddings[key] = embedding


def get_layer(
    input_size: int,
    output_size: int,
    dropout: float = 0.0,
    use_layer_norm: bool = False,
) -> T.nn.Module:
    module_list: list[T.nn.Module] = []
    module_list.append(T.nn.Linear(input_size, output_size))
    if use_layer_norm:
        module_list.append(T.nn.LayerNorm(output_size))
    module_list.append(T.nn.LeakyReLU())
    if dropout > 0.0:
        module_list.append(T.nn.Dropout(dropout))
    return T.nn.Sequential(*module_list)


class Model(T.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        latent_size: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        threshold: float = 0.5,
    ):
        super().__init__()

        self.threshold = threshold

        self.model = T.nn.Sequential(
            get_layer(input_size, latent_size, dropout=dropout, use_layer_norm=use_layernorm),
            get_layer(latent_size, latent_size, dropout=dropout, use_layer_norm=use_layernorm),
            get_layer(latent_size, latent_size, dropout=dropout, use_layer_norm=use_layernorm),

            T.nn.Linear(latent_size, output_size),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.model(x)

    def to_prob(self, x: T.Tensor) -> T.Tensor:
        x = T.sigmoid(T.pow(T.e, x))
        return T.where(x > self.threshold, T.tensor(1), T.tensor(0))


@dataclass
class EvalResult:
    acc: float
    prec: float
    rec: float
    f1: float
    loss: float


def evaluate_model(
    model: T.nn.Module,
    dl_val: DataLoader,
    criterion: T.nn.Module,
) -> EvalResult:
    with T.no_grad():
        TPs: int = 0
        FPs: int = 0
        TNs: int = 0
        FNs: int = 0
        val_loss = 0.0
        for X_batch, y_batch in dl_val:
            X_batch, y_batch = X_batch.to(T.device('cuda')), y_batch

            y_hat = model(X_batch).cpu()
            val_loss += criterion(y_hat, y_batch.float()).item()

            y_pred = model.to_prob(y_hat)  # type: ignore

            TPs += int(T.sum(y_pred * y_batch).item())
            FPs += int(T.sum(y_pred * (1 - y_batch)).item())
            TNs += int(T.sum((1 - y_pred) * (1 - y_batch)).item())
            FNs += int(T.sum((1 - y_pred) * y_batch).item())

        acc: float = (TPs + TNs) / (TPs + TNs + FPs + FNs)

        if TPs + FPs == 0:
            prec = 0.0
        else:
            prec = TPs / (TPs + FPs)

        if TPs + FNs == 0:
            rec = 0.0
        else:
            rec = TPs / (TPs + FNs)

        if 2 * TPs + FPs + FNs == 0:
            f1 = 0.0
        else:
            f1 = 2 * TPs / (2 * TPs + FPs + FNs)

    return EvalResult(
        acc=acc,
        prec=prec,
        rec=rec,
        f1=f1,
        loss=val_loss,
    )


@dataclass
class TrainingResult:
    model: T.nn.Module
    train_result: EvalResult
    eval_result: EvalResult
    best_loss: float
    best_epoch: int


@dataclass
class TrainingParams:
    latent_size: int
    lr: float
    dropout: float
    weight_decay: float | None = None
    pos_weights: T.Tensor | None = None
    use_layernorm: bool = False
    threshold: float = 0.5


def train_neural_network(
    ds_train: Dataset,
    ds_val: Dataset,
    params: TrainingParams,
    output_size: int,
    patience: int = 5,
    max_epochs: int = 100,
) -> TrainingResult:
    model = Model(
        input_size=ds_train[0][0].shape[0],
        output_size=output_size,
        latent_size=params.latent_size,
        dropout=params.dropout if params.dropout is not None else 0.0,
        use_layernorm=params.use_layernorm,
        threshold=params.threshold,
    ).to(T.device('cuda'))

    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)

    crit_params = {}
    if params.pos_weights is not None:
        crit_params['pos_weight'] = params.pos_weights
    criterion = T.nn.BCEWithLogitsLoss(**crit_params)  # type: ignore

    opt_params = {}
    if params.lr is not None:
        opt_params['lr'] = params.lr
    if params.weight_decay is not None:
        opt_params['weight_decay'] = params.weight_decay
    optimizer = T.optim.AdamW(
        model.parameters(),
        **opt_params,  # type: ignore
    )

    best_loss = float('inf')
    best_model = model
    best_epoch = -1
    remaining_patience = patience
    for epoch in tqdm.trange(max_epochs, desc="Training NN", leave=False):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in dl_train:
            X_batch, y_batch = X_batch.to(T.device('cuda')), y_batch

            optimizer.zero_grad()
            y_pred = model(X_batch).cpu()
            loss = criterion(y_pred, y_batch.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with T.no_grad():
            val_res = evaluate_model(model, dl_val, criterion)

        if val_res.loss < best_loss:
            best_loss = val_res.loss
            best_model = deepcopy(model)
            best_epoch = epoch
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            break

    model = best_model
    model.eval()
    return TrainingResult(
        train_result=evaluate_model(model, dl_train, criterion),
        eval_result=evaluate_model(model, dl_val, criterion),
        model=model.cpu(),
        best_loss=best_loss,
        best_epoch=best_epoch,
    )


@dataclass
class CvResult:
    best_epochs: list[int]

    val_loss: list[float]
    val_accs: list[float]
    val_precs: list[float]
    val_recs: list[float]
    val_f1s: list[float]

    train_loss: list[float]
    train_accs: list[float]
    train_precs: list[float]
    train_recs: list[float]
    train_f1s: list[float]


def train_cv(
    trainer: t.Callable[[Dataset, Dataset], TrainingResult],
    dataset: Dataset,
    n_splits: int,
    random_state: int = 42,
) -> CvResult:
    results: list[TrainingResult] = []

    kfolder = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = list(range(len(dataset)))  # type: ignore
    folds = kfolder.split(indices)  # type: ignore
    for train_idx, val_idx in tqdm.tqdm(folds, total=n_splits, desc="Cross-validation"):
        train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
        ds_train, ds_val = Subset(dataset, train_idx), Subset(dataset, val_idx)
        results.append(trainer(ds_train=ds_train, ds_val=ds_val))  # type: ignore

    return CvResult(
        best_epochs=[r.best_epoch for r in results],

        val_loss=[r.eval_result.loss for r in results],
        val_accs=[r.eval_result.acc for r in results],
        val_precs=[r.eval_result.prec for r in results],
        val_recs=[r.eval_result.rec for r in results],
        val_f1s=[r.eval_result.f1 for r in results],

        train_loss=[r.train_result.loss for r in results],
        train_accs=[r.train_result.acc for r in results],
        train_precs=[r.train_result.prec for r in results],
        train_recs=[r.train_result.rec for r in results],
        train_f1s=[r.train_result.f1 for r in results],
    )


def objective(
    trial: optuna.Trial,
    embedding_registry: EmbeddingRegistry,
    output_size: int = 104,
    n_splits: int = 5,
) -> float:
    embedding_key: str = trial.suggest_categorical(
        'embedding_key',
        list(embedding_registry.embeddings.keys()),
    )
    X, y = embedding_registry.embeddings[embedding_key].get_data(n_classes=output_size)
    y = y[:, 1:]

    if trial.suggest_categorical('use_pos_weights', [True, False]):
        pos_weights = T.tensor([1.0 / (y[:, i].sum().item() + 1) for i in range(y.shape[1])], dtype=T.float32)
    else:
        pos_weights = None

    ds = TensorDataset(X, y)
    trainer = partial(
        train_neural_network,
        params=TrainingParams(
            lr=trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            dropout=trial.suggest_float('dropout', 0.0, 0.9),
            latent_size=trial.suggest_int('latent_size', 8, 32),
            pos_weights=pos_weights,
            use_layernorm=trial.suggest_categorical('use_layernorm', [True, False]),
            threshold=trial.suggest_float('threshold', 0.0, 1.0),
        ),
        output_size=y.shape[1],
    )

    res = train_cv(trainer=trainer, dataset=ds, n_splits=n_splits)

    trial.set_user_attr('f1_val', np.mean(res.val_f1s))
    trial.set_user_attr('f1_train', np.mean(res.train_f1s))

    trial.set_user_attr('prec_val', np.mean(res.val_precs))
    trial.set_user_attr('prec_train', np.mean(res.train_precs))

    trial.set_user_attr('rec_val', np.mean(res.val_recs))
    trial.set_user_attr('rec_train', np.mean(res.train_recs))

    trial.set_user_attr('acc_val', np.mean(res.val_accs))
    trial.set_user_attr('acc_train', np.mean(res.train_accs))

    trial.set_user_attr('best_epoch', np.mean(res.best_epochs))
    trial.set_user_attr('val_loss', np.mean(res.val_loss))
    trial.set_user_attr('train_loss', np.mean(res.train_loss))

    return float(np.mean(res.val_loss))


def main(
    dataset_path: str,
    models: list[ClipModelLoadConfig],
    study_folder: Path,
    study_name: str,
):
    embedding_registry = EmbeddingRegistry(types={'train'})
    for model in models:
        for type_ in embedding_registry.registered_types:
            path = Path(dataset_path.format(type_=type_))
            embedding_registry.register(EmbeddingRegistry.load(EmbeddingLoadConfig(
                model_name=model.model_name,
                model_weights=model.model_weights,
                type_=type_,
                path=path,
            )))

    study = optuna.create_study(
        direction=optuna.study.StudyDirection.MINIMIZE,
        study_name=f"{study_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        storage=f'sqlite:///{str(study_folder)}/{study_name}.db',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
            group=True,
        ),
    )

    study.optimize(
        partial(
            objective,
            embedding_registry=embedding_registry,
            n_splits=5,
        ),
        n_trials=100,
    )

    params = study.best_params
    X, y = embedding_registry.embeddings[params['embedding_key']].get_data(n_classes=104)
    y = y[:, 1:]

    ds = TensorDataset(X, y)
    split_idx = int(0.8 * len(ds))
    ds_train, ds_val = random_split(ds, [split_idx, len(ds) - split_idx])

    y_train = ds_train[:][1]
    pos_weights = T.tensor([1.0 / (y_train[:, i].sum().item() + 1) for i in range(y.shape[1])], dtype=T.float32)

    res = train_neural_network(
        ds_train=ds_train,
        ds_val=ds_val,
        params=TrainingParams(
            lr=params['lr'],
            dropout=params['dropout'],
            latent_size=params['latent_size'],
            pos_weights=pos_weights,
            use_layernorm=params['use_layernorm'],
        ),
        output_size=y.shape[1],
    )

    embedding_registry = EmbeddingRegistry(types={'test'})
    for model in models:
        for type_ in embedding_registry.registered_types:
            path = Path(dataset_path.format(type_=type_))
            embedding_registry.register(EmbeddingRegistry.load(EmbeddingLoadConfig(
                model_name=model.model_name,
                model_weights=model.model_weights,
                type_=type_,
                path=path,
            )))

    embedding_key = params['embedding_key'].replace("type_='train'", "type_='test'")
    X, y = embedding_registry.embeddings[embedding_key].get_data(n_classes=104)
    y = y[:, 1:]

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    y = ds[:][1]
    pos_weights = T.tensor([
        (y.shape[0] - y[:, i].sum().item()) / y[:, i].sum().item()
        for i in range(103)
    ], dtype=T.float32)

    eval_res = evaluate_model(
        model=res.model.cuda(),
        dl_val=dl,
        criterion=T.nn.BCEWithLogitsLoss(pos_weight=pos_weights),
    )

    print("--- Test results ---")
    print(f"Accuracy:  {eval_res.acc:.4f}")
    print(f"Precision: {eval_res.prec:.4f}")
    print(f"Recall:    {eval_res.rec:.4f}")
    print(f"F1:        {eval_res.f1:.4f}")
    print(f"Val loss:  {eval_res.loss:.4f}")

    baseline_model = Model(
        input_size=ds[0][0].shape[0],
        output_size=103,
        latent_size=params['latent_size'],
        dropout=params['dropout'],
        use_layernorm=params['use_layernorm'],
        threshold=params['threshold'],
    ).to(T.device('cuda'))

    eval_res = evaluate_model(
        model=baseline_model.cuda(),
        dl_val=dl,
        criterion=T.nn.BCEWithLogitsLoss(pos_weight=pos_weights),
    )

    print("--- Baseline ---")
    print(f"Accuracy:  {eval_res.acc:.4f}")
    print(f"Precision: {eval_res.prec:.4f}")
    print(f"Recall:    {eval_res.rec:.4f}")
    print(f"F1:        {eval_res.f1:.4f}")
    print(f"Val loss:  {eval_res.loss:.4f}")

    print(eval_res)


if __name__ == '__main__':
    main(
        dataset_path='data/FoodSeg103/processed_{type_}',
        models=[
            ClipModelLoadConfig('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
            ClipModelLoadConfig('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'),
            ClipModelLoadConfig('RN101', 'openai'),
            ClipModelLoadConfig('RN50', 'openai'),
            ClipModelLoadConfig('RN50x64', 'openai'),
            ClipModelLoadConfig('ViT-B-32', 'laion2b_s34b_b79k'),
            ClipModelLoadConfig('ViT-H-14', 'laion2b_s32b_b79k'),
        ],
        study_folder=Path('studies/classifiers/'),
        study_name='classifier_neural_networks',
    )
