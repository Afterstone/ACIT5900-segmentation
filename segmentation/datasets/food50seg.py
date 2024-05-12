from __future__ import annotations

import pickle
import shutil
import subprocess
import tempfile
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
import PIL.Image as PIL_Image  # type: ignore
import torch as T
import torchvision.transforms as TV_trfs  # type: ignore
from tqdm import tqdm  # type: ignore

import segmentation.config as config

from ._base import (AbstractSegmentationDataset, download_file,
                    get_deterministic_permutation)


class Food50Seg(AbstractSegmentationDataset):
    def __init__(self, _load_check: bool = True):
        super().__init__(_load_check=_load_check)

        if _load_check:
            raise Exception("Please load data using the UECFoodPixComplete.load_* methods")

    @classmethod
    def load_data(
        cls,
        root: Path,
        tensor_size: t.Tuple[int, int] = (256, 256),
        split_name: str | None = None,
        load_first_n: int | None = None,
        load_subset_by_index: set[int] | None = None,
    ) -> AbstractSegmentationDataset:
        # TODO: Implement me
        raise NotImplementedError
        # dataset = cls(_load_check=False)

        # if not split_name in ['train', 'test']:
        #     raise ValueError('split_name must be either "train" or "test"')
        # dataset.split_name = split_name

        # category_file = root / 'category.txt'
        # df_category = pd.read_csv(category_file, sep='\t', header=0, names=['id', 'name'])
        # dataset.categories = {int(row['id']): row['name'] for _, row in df_category.iterrows()}

        # image_folder = root / split_name / 'img'
        # dataset.image_paths = sorted(list(image_folder.glob('*.jpg')))
        # if load_first_n is not None:
        #     dataset.image_paths = dataset.image_paths[:load_first_n]
        # elif load_subset_by_index is not None:
        #     dataset.image_paths = [dataset.image_paths[i] for i in load_subset_by_index]

        # annotations_folder = root / split_name / 'mask'
        # dataset.annotations_paths = sorted(list(annotations_folder.glob('*.png')))
        # if load_first_n is not None:
        #     dataset.annotations_paths = dataset.annotations_paths[:load_first_n]
        # elif load_subset_by_index is not None:
        #     dataset.annotations_paths = [dataset.annotations_paths[i] for i in load_subset_by_index]

        # ip = [x.stem for x in dataset.image_paths]
        # ap = [x.stem for x in dataset.annotations_paths]
        # if len(ip) == 0 or len(ip) != len(ap) or ip != ap:
        #     raise ValueError('Image and annotation files do not match')

        # Xs = []
        # ys = []

        # to_tensor = TV_trfs.ToTensor()
        # progdesc = 'Loading images and annotations. #Errors: {}'
        # progbar = tqdm(range(len(dataset.image_paths)), desc=progdesc.format(0))
        # errors = []
        # for img_path, ann_path in zip(dataset.image_paths, dataset.annotations_paths):
        #     progbar.update(1)
        #     with PIL_Image.open(img_path) as img, PIL_Image.open(ann_path) as ann:
        #         try:
        #             if img.size != ann.size:
        #                 raise ValueError(
        #                     f'Image and annotation sizes do not match, img: {img.size}, ann: {ann.size}')

        #             if img.mode != 'RGB':
        #                 img = img.convert('RGB')
        #             if img.size != tensor_size:
        #                 img = img.resize(tensor_size, PIL_Image.BILINEAR)

        #             if img.mode != 'L':
        #                 ann = ann.convert('L')
        #             if ann.size != tensor_size:
        #                 ann = ann.resize(tensor_size, PIL_Image.NEAREST)

        #             Xs.append(to_tensor(img).permute(1, 2, 0))
        #             ys.append(T.from_numpy(np.array(ann, dtype=np.uint8)))
        #         except Exception as e:
        #             errors.append((img_path, ann_path, e))
        #             progbar.set_description(progdesc.format(len(errors)))
        # if len(errors) > 0:
        #     print('Errors encountered:')
        #     for img_path, ann_path, err in errors:
        #         print(f'  {img_path} - {ann_path}: {err}')
        #     print()

        # dataset.X = T.stack(Xs)
        # dataset.y = T.stack(ys)

        # return dataset

    @classmethod
    def load_pickle(cls, folder: Path) -> AbstractSegmentationDataset:
        # TODO: Implement me
        raise NotImplementedError
        # dataset = cls(_load_check=False)
        # if not folder.exists():
        #     raise ValueError(f'Folder {folder} does not exist')

        # pickle_path = folder / 'metadata.pkl'
        # if not pickle_path.exists():
        #     raise ValueError(f'File {pickle_path} does not exist')

        # with open(pickle_path, 'rb') as f:
        #     metadata = pickle.load(f)
        # dataset.categories = metadata['categories']
        # dataset.image_paths = metadata['image_paths']
        # dataset.annotations_paths = metadata['annotations_paths']
        # dataset.X = T.load(pickle_path.parent / metadata['X_path'])
        # dataset.y = T.load(pickle_path.parent / metadata['y_path'])

        # return dataset

    def dump_pickle(self, folder: Path) -> None:
        # TODO: Implement me
        raise NotImplementedError
        # folder.mkdir(parents=True, exist_ok=True)
        # metadata = {
        #     'categories': self.categories,
        #     'image_paths': self.image_paths,
        #     'annotations_paths': self.annotations_paths,
        #     'X_path': f'X_{self.split_name}.pt',
        #     'y_path': f'y_{self.split_name}.pt',
        # }
        # T.save(self.X, folder / metadata['X_path'])  # type: ignore
        # T.save(self.y, folder / metadata['y_path'])  # type: ignore
        # with open(folder / 'metadata.pkl', 'wb') as f:
        #     pickle.dump(metadata, f)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> t.Tuple[T.Tensor, T.Tensor]:
        return self.X[idx], self.y[idx]

    def get_cat_ids_from_annotation_masks(self, annotation_masks: T.Tensor) -> list[int]:
        # TODO: Implement me
        raise NotImplementedError
        # cat_ids = annotation_masks.ravel().unique().squeeze()
        # return [int(i) for i in cat_ids]

    def get_sparse_annotation_masks(self, annotation_masks: T.Tensor) -> dict[int, T.Tensor]:
        # TODO: Implement me
        raise NotImplementedError
        # annotation_masks = annotation_masks[:, :, :, :3]
        # masks: dict[int, T.Tensor] = {}
        # indices = [int(i) for i in self.get_cat_ids_from_annotation_masks(annotation_masks)]
        # for i in indices:
        #     masks[i] = T.sum((annotation_masks == i).float(), dim=3)
        # return masks


def main(
    dest_dir: Path = Path('./data/'),
    temp_dir: Path = Path('./temp/'),
):
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)

    imgs_zip_filename = "50food.zip"
    imgs_zip_path = temp_dir / imgs_zip_filename
    if not imgs_zip_path.exists():
        download_file(
            url='https://www.cmlab.csie.ntu.edu.tw/project/food/file/50data.zip',
            destination=temp_dir,
            filename=imgs_zip_filename,
        )

    masks_zip_filename = "Benchmarking-segmentation-algorithms-dataset.zip"
    masks_zip_path = temp_dir / masks_zip_filename
    if not masks_zip_path.exists():
        download_file(
            url='http://www.ivl.disco.unimib.it/download/Benchmarking-segmentation-algorithms-dataset.zip',
            destination=temp_dir,
            filename=masks_zip_filename,
        )

    dest_dir_uecfpc = dest_dir / 'Food50Seg'
    if not dest_dir_uecfpc.exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = subprocess.run(f"7z x {str(masks_zip_path)} -p'{config.FOOD50SEG_PASSWORD}' -o'{str(tmp_dir)}/'", shell=True, check=True)
            if res.returncode != 0:
                raise ValueError(f'Error unzipping {masks_zip_path}')

            dest_dir_uecfpc_masks = dest_dir_uecfpc / 'masks'
            dest_dir_uecfpc_masks.mkdir(parents=True, exist_ok=True)

            tmp_dir_path = Path(tmp_dir)
            ...
            # TODO: Unzip the files inside the masks folder

        with tempfile.TemporaryDirectory() as tmp_dir:
            res = subprocess.run(['unzip', str(imgs_zip_path), '-d', str(tmp_dir)], check=True)
            if res.returncode != 0:
                raise ValueError(f'Error unzipping {imgs_zip_path}')

            dest_dir_uecfpc_imgs = dest_dir_uecfpc / 'images'
            dest_dir_uecfpc_imgs.mkdir(parents=True, exist_ok=True)

            tmp_dir_path = Path(tmp_dir)
            for p in tmp_dir_path.iterdir():
                shutil.move(p, dest_dir_uecfpc_imgs)

        print()

    # ds_test = Food50Seg.load_data(dest_dir_uecfpc, split_name='test')
    # ds_test.dump_pickle(dest_dir_uecfpc / 'processed_test')
    # del ds_test
    # ds_test = Food50Seg.load_pickle(dest_dir_uecfpc / 'processed_test')
    # del ds_test

    # ds_train = Food50Seg.load_data(dest_dir_uecfpc, split_name='train')
    # ds_train.dump_pickle(dest_dir_uecfpc / 'processed_train')
    # del ds_train
    # ds_train = Food50Seg.load_pickle(dest_dir_uecfpc / 'processed_train')
    # del ds_train


if __name__ == '__main__':
    main()
