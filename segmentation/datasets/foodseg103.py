from __future__ import annotations

import pickle
import shutil
import subprocess
import tempfile
import typing as t
from calendar import c
from curses import meta
from json import load
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import PIL.Image as PIL_Image  # type: ignore
import requests  # type: ignore
import torch as T
import torchvision.transforms as TV_trfs  # type: ignore
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore


class SegFoodDataset(Dataset):
    def __init__(self, _load_check: bool = True):
        super().__init__()

        if _load_check:
            raise Exception("Please load data using the SegFoodDataset.load_* methods")

        self.split_name: str  # type: ignore
        self.category_df: pd.DataFrame  # type: ignore
        self.image_paths: t.List[Path]  # type: ignore
        self.annotations_paths: t.List[Path]  # type: ignore
        self.img_sizes: t.List[t.Tuple[int, int]]  # type: ignore
        self.X: T.Tensor = T.tensor([])  # type: ignore
        self.y: T.Tensor = T.tensor([])  # type: ignore

    @classmethod
    def load_data(
        cls,
        root: Path,
        train: bool = True,
        load_first_n: int | None = None,
        tensor_size: t.Tuple[int, int] = (256, 256),
    ) -> SegFoodDataset:
        dataset = cls(_load_check=False)
        dataset.split_name = 'train' if train else 'test'

        category_file = root / 'category_id.txt'
        dataset.category_df = pd.read_csv(category_file, sep='\t', header=None, names=['id', 'category'])

        image_folder = root / 'Images' / 'img_dir' / dataset.split_name
        dataset.image_paths = sorted(list(image_folder.glob('*.jpg')))
        if load_first_n is not None:
            dataset.image_paths = dataset.image_paths[:load_first_n]

        annotations_folder = root / 'Images' / 'ann_dir' / dataset.split_name
        dataset.annotations_paths = sorted(list(annotations_folder.glob('*.png')))
        if load_first_n is not None:
            dataset.annotations_paths = dataset.annotations_paths[:load_first_n]

        ip = [x.stem for x in dataset.image_paths]
        ap = [x.stem for x in dataset.annotations_paths]
        if len(ip) == 0 or len(ip) != len(ap) or ip != ap:
            raise ValueError('Image and annotation files do not match')

        dataset.img_sizes = []
        Xs = []
        ys = []

        to_tensor = TV_trfs.ToTensor()
        progdesc = 'Loading images and annotations. #Errors: {}'
        progbar = tqdm(range(len(dataset.image_paths)), desc=progdesc.format(0))
        errors = []
        for img_path, ann_path in zip(dataset.image_paths, dataset.annotations_paths):
            progbar.update(1)
            with PIL_Image.open(img_path) as img, PIL_Image.open(ann_path) as ann:
                try:
                    dataset.img_sizes.append(img.size)

                    if img.size != ann.size:
                        raise ValueError(
                            f'Image and annotation sizes do not match, img: {img.size}, ann: {ann.size}')

                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if img.size != tensor_size:
                        img = img.resize(tensor_size, PIL_Image.BILINEAR)

                    if img.mode != 'L':
                        ann = ann.convert('L')
                    if ann.size != tensor_size:
                        ann = ann.resize(tensor_size, PIL_Image.NEAREST)

                    Xs.append(to_tensor(img).permute(1, 2, 0))
                    ys.append(T.from_numpy(np.array(ann, dtype=np.uint8)))
                except Exception as e:
                    errors.append((img_path, ann_path, e))
                    progbar.set_description(progdesc.format(len(errors)))
        if len(errors) > 0:
            print('Errors encountered:')
            for img_path, ann_path, err in errors:
                print(f'  {img_path} - {ann_path}: {err}')
            print()

        dataset.X = T.stack(Xs)
        dataset.y = T.stack(ys)

        return dataset

    @classmethod
    def load_pickle(cls, folder: Path) -> SegFoodDataset:
        dataset = cls(_load_check=False)
        if not folder.exists():
            raise ValueError(f'Folder {folder} does not exist')

        pickle_path = folder / 'metadata.pkl'
        if not pickle_path.exists():
            raise ValueError(f'File {pickle_path} does not exist')

        with open(pickle_path, 'rb') as f:
            metadata = pickle.load(f)
        dataset.category_df = metadata['category_df']
        dataset.image_paths = metadata['image_paths']
        dataset.annotations_paths = metadata['annotations_paths']
        dataset.img_sizes = metadata['img_sizes']
        dataset.X = T.load(pickle_path.parent / metadata['X_path'])
        dataset.y = T.load(pickle_path.parent / metadata['y_path'])

        return dataset

    def dump_pickle(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {
            'category_df': self.category_df,
            'image_paths': self.image_paths,
            'annotations_paths': self.annotations_paths,
            'img_sizes': self.img_sizes,
            'X_path': f'X_{self.split_name}.pt',
            'y_path': f'y_{self.split_name}.pt',
        }
        T.save(self.X, folder / metadata['X_path'])
        T.save(self.y, folder / metadata['y_path'])
        with open(folder / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> t.Tuple[T.Tensor, T.Tensor]:
        return self.X[idx], self.y[idx]


def _download(
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
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()


def main(
    dest_dir: Path = Path('./data/'),
    temp_dir: Path = Path('./temp/'),
    password: str = 'LARCdataset9947',
):
    zip_filename = "FoodSeg103.zip"
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)
        _download(
            url='https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip',
            destination=temp_dir,
            filename=zip_filename,
        )

    dest_dir_foodseg = dest_dir / 'FoodSeg103'
    if not dest_dir_foodseg.exists():
        zip_location = temp_dir / zip_filename
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = subprocess.run(
                ['unzip', '-P', password, str(zip_location), '-d', str(tmp_dir)],
                check=True,
            )
            if res.returncode != 0:
                raise ValueError(f'Error unzipping {zip_location}')

            shutil.move(Path(tmp_dir) / 'FoodSeg103', dest_dir)
        print()

    ds_train = SegFoodDataset.load_data(dest_dir_foodseg, train=True)
    ds_train.dump_pickle(dest_dir_foodseg / 'processed_train')
    del ds_train
    ds_train = SegFoodDataset.load_pickle(dest_dir_foodseg / 'processed_train')
    del ds_train

    ds_test = SegFoodDataset.load_data(dest_dir_foodseg, train=False)
    ds_test.dump_pickle(dest_dir_foodseg / 'processed_test')
    del ds_test
    ds_test = SegFoodDataset.load_pickle(dest_dir_foodseg / 'processed_test')


if __name__ == '__main__':
    main()
