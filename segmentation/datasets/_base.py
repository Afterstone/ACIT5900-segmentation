from __future__ import annotations

import typing as t
from abc import abstractmethod
from pathlib import Path

import requests
import torch as T
import tqdm
from torch.utils.data import Dataset


def get_deterministic_permutation(n: int, seed: int = 42) -> set[int]:
    generator = T.Generator().manual_seed(seed)
    permutation = set(T.randperm(n, generator=generator).tolist())
    return permutation


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 1024,
    filename: str | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if filename is None:
        save_path = destination / url.split('/')[-1]
    else:
        save_path = destination / filename

    with open(save_path, 'wb') as fd:
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        progress_bar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()


class AbstractSegmentationDataset(Dataset):
    """This class defines the interface for segmentation datasets.

    New datasets should be created in ./segmentation/datasets as a standalone Python file.
    The file should contain logic to download and preprocess data, if possible.
    Data should then be loaded with the load_data class method and dumped as part of the
    setup process. (See ./segmentation/datasets/foodseg103.py for an example.)

    The dataset class should inherit from this class and implement the abstract methods.

    For now, the class has some suggested class members that should probably be refactored
    soon.

    Instance parameters:
    - split_name (str): Used to label the split. Might optionally be used to determine where
        to load data from.
    - categories (dict[int | str, str]: Maps category IDs to category names. Categories are the
        text label for each class in the dataset.
    - image_paths (List[Path]): Paths to the images in the dataset. Assumes the original data
        is stored in image folders loadable by PIL.Image.open.
    - annotations_paths (List[Path]): Paths to the annotations in the dataset. Assumes the annotation
        masks are stored as PNG images with the same index as the corresponding image. Must be loadable
        by PIL.Image.open.
    - X (T.Tensor): The input data tensor. Should be a 4D tensor with shape (N, C, H, W), where N is the
        number of samples, C is the number of channels, and H, W are the height and width of the images.
    - Y (T.Tensor): The target data tensor. Should be a 3D tensor with shape (N, C, H, W), where N is the
        number of samples, C is the number of classes, and H, W are the height and width of the images.
    """

    def __init__(self, _load_check: bool = True):
        super().__init__()

        if _load_check:
            raise Exception("Please load data using the Class.load_* methods")

        self.split_name: str  # type: ignore
        self.categories: dict[int | str, str]  # type: ignore
        self.image_paths: t.List[Path]  # type: ignore
        self.annotations_paths: t.List[Path]  # type: ignore
        self.X: T.Tensor = T.tensor([])  # type: ignore
        self.y: T.Tensor = T.tensor([])  # type: ignore

    @abstractmethod
    def load_data(
        cls,
        root: Path,
        tensor_size: t.Tuple[int, int],
        split_name: str | None = None,
        load_first_n: int | None = None,
        load_subset_by_index: set[int] | None = None,
    ) -> 'AbstractSegmentationDataset':
        raise NotImplementedError

    @abstractmethod
    def load_pickle(cls, folder: Path) -> 'AbstractSegmentationDataset':
        raise NotImplementedError

    def dump_pickle(self, folder: Path) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> t.Tuple[T.Tensor, T.Tensor]:
        raise NotImplementedError
