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
from torch.utils.data import (DataLoader, Dataset, TensorDataset,  # noqa: E402
                              random_split)

from segmentation.datasets.foodseg103 import FoodSegDataset
from segmentation.models import get_clip_model

# def eval_model(
#     model: T.nn.Module,
#     dl_test: DataLoader,
#     threshold: float = 0.1,
# ):
#     metrics: dict[str, float] = defaultdict()
#     model.eval()
#     with T.no_grad():
#         TPs = 0
#         FPs = 0
#         TNs = 0
#         FNs = 0
#         for X_batch, y_batch in dl_test:
#             X_batch, y_batch = X_batch.to(T.device('cuda')), y_batch

#             y_pred = model(X_batch).cpu()

#             y_pred = T.sigmoid(y_pred)
#             y_pred = T.where(y_pred > threshold, T.tensor(1), T.tensor(0))

#             TPs += T.sum(y_pred * y_batch).item()
#             FPs += T.sum(y_pred * (1 - y_batch)).item()
#             TNs += T.sum((1 - y_pred) * (1 - y_batch)).item()
#             FNs += T.sum((1 - y_pred) * y_batch).item()

#         metrics['acc'] = (TPs + TNs) / (TPs + TNs + FPs + FNs)

#         if TPs + FPs == 0:
#             metrics['prec'] = 0
#         else:
#             metrics['prec'] = TPs / (TPs + FPs)

#         if TPs + FNs == 0:
#             metrics['rec'] = 0
#         else:
#             metrics['rec'] = TPs / (TPs + FNs)

#         if 2 * TPs + FPs + FNs == 0:
#             metrics['f1'] = 0
#         else:
#             metrics['f1'] = 2 * TPs / (2 * TPs + FPs + FNs)

#     return metrics


# def train_classifier(
#     X: T.Tensor,
#     image_classes: list[list[int]],
#     n_splits: int = 5,
#     n_epochs: int = 50,
#     patience: int = 5,
#     n_threshold_steps: int = 30,
# ) -> tuple[T.nn.Module, dict[str, list[float]]]:
#     X = X.to(T.float32)
#     y = T.zeros((len(image_classes), 104), dtype=T.long)

#     counter = Counter()
#     for i, classes in enumerate(image_classes):
#         for c in classes:
#             y[i, c] = 1
#             counter[c] += 1

#     # class_weights = T.tensor([1.0 / (counter[i]) for i in range(104)], dtype=T.float32)
#     class_weights = T.tensor([counter[i] for i in range(104)], dtype=T.float32)
#     criterion = T.nn.BCEWithLogitsLoss(
#         # pos_weight=class_weights
#         # weight=class_weights
#     )

#     global_best_model: T.nn.Module = None  # type: ignore
#     global_best_loss = float('inf')

#     metrics: dict[str, list[float]] = defaultdict(list)
#     kfolder = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     for train_idx, val_idx in tqdm.tqdm(kfolder.split(X), desc="Training classifier", total=n_splits):  # type: ignore
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         ds_train = TensorDataset(X_train, y_train)
#         dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)

#         ds_val = TensorDataset(X_val, y_val)
#         dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)

#         model = Model(X.shape[1], 104).to(T.device('cuda'))
#         optimizer = T.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

#         best_loss = float('inf')
#         best_model = model
#         best_epoch = -1
#         remaining_patience = patience
#         for epoch in range(n_epochs):
#             model.train()
#             train_loss = 0.0
#             for X_batch, y_batch in dl_train:
#                 X_batch, y_batch = X_batch.to(T.device('cuda')), y_batch

#                 optimizer.zero_grad()
#                 y_pred = model(X_batch).cpu()
#                 loss = criterion(y_pred, y_batch.float())
#                 loss.backward()
#                 optimizer.step()

#                 train_loss += loss.item()

#             model.eval()
#             with T.no_grad():
#                 val_loss = 0.0
#                 for X_batch, y_batch in dl_val:
#                     X_batch, y_batch = X_batch.to(T.device('cuda')), y_batch

#                     y_pred = model(X_batch).cpu()
#                     loss = criterion(y_pred, y_batch.float())
#                     val_loss += loss.item()

#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 best_model = deepcopy(model)
#                 best_epoch = epoch
#                 remaining_patience = patience
#             else:
#                 remaining_patience -= 1

#             if remaining_patience < 0:
#                 break

#             if val_loss < global_best_loss:
#                 global_best_loss = val_loss
#                 global_best_model = deepcopy(model)

#         metrics['val_losses'].append(best_loss)
#         metrics['best_epoch'].append(best_epoch)

#     model = global_best_model
#     model.eval()
#     progbar = tqdm.tqdm(total=n_threshold_steps, desc="Evaluating model")
#     for threshold in np.linspace(0.0, 1.0, n_threshold_steps):
#         m = eval_model(model, dl_val, threshold)
#         for k, v in m.items():
#             metrics[f"{k}_{threshold}"].append(v)
#         progbar.update(1)
#     progbar.close()

#     return best_model, metrics


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


def get_layer(input_size: int, output_size: int, dropout: float = 0.0) -> T.nn.Module:
    return T.nn.Sequential(
        T.nn.Linear(input_size, output_size),
        T.nn.LayerNorm(output_size),
        T.nn.LeakyReLU(),
        T.nn.Dropout(dropout),
    )


class Model(T.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.model = T.nn.Sequential(
            get_layer(input_size, 128, dropout=0.2),
            get_layer(128, 128, dropout=0.2),
            get_layer(128, 128, dropout=0.2),

            T.nn.Linear(128, output_size),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.model(x)

    def predict(self, x: T.Tensor) -> T.Tensor:
        return F.sigmoid(self(x))


# @dataclass
# class TrainingResult:
#     # TODO: Finish me.
#     ...


# def train_neural_network() -> TrainingResult:
#     # TODO: Finish me.
#     ...


@dataclass
class CvResult:
    # TODO: Finish me.
    acc: list[float]
    prec: list[float]
    rec: list[float]
    f1: list[float]


# def train_cv(
#     X: T.Tensor,
#     y: T.Tensor,
#     n_splits: int,
#     random_state: int = 42,
# ) -> CvResult:
#     # TODO: Finish me.

#     kfolder = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     for train_idx, val_idx in tqdm.tqdm(kfolder.split(X), total=n_splits):  # type: ignore
#         # train_neural_network()


def objective(
    trial: optuna.Trial,
    embedding_registry: EmbeddingRegistry,
) -> float:
    # TODO: Finish me.
    # embedding_key: str = trial.suggest_categorical(
    #     'embedding_key',
    #     list(embedding_registry.embeddings.keys()),
    # )
    # X, y = embedding_registry.embeddings[embedding_key].get_data(n_classes=104)

    # results = train_cv(X, y, n_splits=5, random_state=42)
    return 0.0


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
        direction='maximize',
        study_name=study_name,
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
        ),
        n_trials=100,
    )


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
