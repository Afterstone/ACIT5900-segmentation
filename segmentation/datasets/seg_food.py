import shutil
import tempfile
import typing as t
import zipfile
from ast import Not
from calendar import c
from os import path
from pathlib import Path
from re import A

import kaggle
import numpy as np
import PIL
import PIL.Image as PIL_Image
import torch as T
import torchvision.transforms as TV_trfs
from torch.utils.data import Dataset
from tqdm import tqdm, trange

IMG_SIZE = (256, 256)


class SegFoodDataset(Dataset):
    def __init__(self, root: Path, load_first_n: int | None = None):
        self.dir = root
        self.images_dir = self.dir / 'images'
        self.masks_dir = self.dir / 'masks'

        self.img_ids = [f.stem for f in self.images_dir.glob('*.pt')]
        if set(self.img_ids) != set([f.stem for f in self.images_dir.glob('*.pt')]):
            raise ValueError(f'Image and mask directories do not contain matching files.')

        self.images_pt_paths = [self.images_dir / f'{name}.pt' for name in self.img_ids]
        self.masks_pt_paths = [self.masks_dir / f'{name}.pt' for name in self.img_ids]
        if load_first_n:
            self.images_pt_paths = self.images_pt_paths[:load_first_n]
            self.masks_pt_paths = self.masks_pt_paths[:load_first_n]

        self.X: T.Tensor = T.empty((len(self.images_pt_paths), 3, *IMG_SIZE), dtype=T.float32)
        self.y: T.Tensor = T.empty((len(self.masks_pt_paths), 1, *IMG_SIZE), dtype=T.float32)

        for i in trange(len(self.images_pt_paths), desc='Loading data...'):
            self.X[i] = T.load(self.images_pt_paths[i])
            self.y[i] = T.load(self.masks_pt_paths[i]).float()

    def __len__(self) -> int:
        return len(self.images_pt_paths)

    def _load_data(self) -> None:
        for i, (img_pt, mask_pt) in enumerate(zip(self.images_pt_paths, self.masks_pt_paths)):
            self.X[i] = T.load(img_pt)
            self.y[i] = T.load(mask_pt)

    def __getitem__(self, idx: int) -> t.Tuple[T.Tensor, T.Tensor]:
        return self.X[idx], self.y[idx]


def _download(dataset_name: str, path: Path) -> None:
    api = kaggle.api
    api.authenticate()
    path.parent.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset=dataset_name, path=path, unzip=False, quiet=False)


def get_dest_file_path(
    file_path: Path,
    dest_dir: Path,
    label: str,
) -> Path:
    if 'train' in file_path.parts[-2]:
        dest_file_dir = dest_dir / 'train' / label
    elif 'valid' in file_path.parts[-2]:
        dest_file_dir = dest_dir / 'test' / label
    else:
        raise ValueError(f'Unexpected folder name: {file_path.parts[-2]}')
    dest_file_dir.mkdir(parents=True, exist_ok=True)
    dest_file_path = dest_file_dir / file_path.name

    return dest_file_path


def process_image(source: Path, dest: Path, skip_existing: bool = True) -> None:
    path_tensor = dest.with_suffix('.pt')
    path_image = dest.with_suffix('.png')
    if skip_existing and path_tensor.exists() and path_image.exists():
        return

    path_tensor.parent.mkdir(parents=True, exist_ok=True)
    path_image.parent.mkdir(parents=True, exist_ok=True)

    transforms = TV_trfs.Compose([TV_trfs.ToTensor()])
    with PIL_Image.open(source) as image:
        image = image.resize(IMG_SIZE)
        image = image.convert('RGB')
        tensor = transforms(image)

        image.save(path_image)
        T.save(tensor, path_tensor)


def process_mask(source: Path, dest: Path, skip_existing: bool = True) -> None:
    path_tensor = dest.with_suffix('.pt')
    path_image = dest.with_suffix('.png')

    if skip_existing and path_tensor.exists() and path_image.exists():
        return

    path_tensor.parent.mkdir(parents=True, exist_ok=True)
    path_image.parent.mkdir(parents=True, exist_ok=True)

    transforms = TV_trfs.Compose([TV_trfs.ToTensor()])
    with PIL_Image.open(source) as raw_image:
        # Convert image to numpy array
        arr = np.array(raw_image)
        assert arr.dtype == np.uint8, f'Unexpected dtype: {arr.dtype}'
        assert arr.max() > 0, "Error: Mask is all black"

        arr[arr > 0] = 255

        image = PIL_Image.fromarray(arr)
        image = image.resize(IMG_SIZE)
        image = image.convert('P')

        tensor = transforms(image)
        tensor = tensor.squeeze()
        assert tensor.shape == IMG_SIZE, f'Unexpected shape: {tensor.shape}'

        image.save(path_image)
        T.save(tensor, path_tensor)


def main(
    dest_dir: Path = Path('./data/'),
    temp_dir: Path = Path('./temp/'),
    label: str = 'seg_food',
):
    temp_dir = Path(temp_dir) / label
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)
        _download(
            dataset_name='shashwatwork/segfood-dataset-for-semantic-food-segmentation',
            path=temp_dir,
        )
    zip_path = next(temp_dir.glob('*.zip'))

    dest_dir = dest_dir / label

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Iterate over the file names
        file_list = zip_ref.namelist()
        file_names = [Path(file_name) for file_name in file_list]

        dest_train_images_dir = dest_dir / 'train' / 'images'
        dest_train_images_dir.mkdir(parents=True, exist_ok=True)
        dest_train_masks_dir = dest_dir / 'train' / 'masks'
        dest_train_masks_dir.mkdir(parents=True, exist_ok=True)
        train_names = [f.stem for f in file_names if 'training' in f.parts[-2]]
        for file_name in tqdm(train_names, desc='Extracting files...'):
            path_train_img_tensor = dest_train_images_dir / f'{file_name}.pt'
            path_train_mask_tensor = dest_train_masks_dir / f'{file_name}.pt'
            if path_train_img_tensor.exists() and path_train_mask_tensor.exists():
                continue
            try:
                with tempfile.TemporaryDirectory() as tempfile_dir, tempfile.TemporaryDirectory() as quarantine_dir:
                    tempfile_dir = Path(tempfile_dir)

                    img_path = f'combinedsegmentationmodel/trainingimages/{file_name}.jpg'
                    mask_path = f'combinedsegmentationmodel/traininglabels/{file_name}.png'
                    zip_ref.extract(img_path, tempfile_dir)
                    zip_ref.extract(mask_path, tempfile_dir)

                    quarantine_dir = Path(quarantine_dir)
                    process_image(source=tempfile_dir / img_path, dest=quarantine_dir / 'imgs' / file_name)
                    process_mask(source=tempfile_dir / mask_path, dest=quarantine_dir / 'masks' / file_name)

                    for fname in quarantine_dir.glob('**/imgs/*.*'):
                        shutil.move(src=fname, dst=dest_train_images_dir / fname.name)
                    for fname in quarantine_dir.glob('**/masks/*.*'):
                        shutil.move(src=fname, dst=dest_train_masks_dir / fname.name)
            except Exception as e:
                print(f'Error processing {file_name}. Error: {e}')

        dest_test_images_dir = dest_dir / 'test' / 'images'
        dest_test_images_dir.mkdir(parents=True, exist_ok=True)
        dest_test_masks_dir = dest_dir / 'test' / 'masks'
        dest_test_masks_dir.mkdir(parents=True, exist_ok=True)
        valid_names = [f.stem for f in file_names if 'validation' in f.parts[-2]]
        for file_name in tqdm(valid_names, desc='Extracting files...'):
            path_test_img_tensor = dest_test_images_dir / f'{file_name}.pt'
            path_test_mask_tensor = dest_test_masks_dir / f'{file_name}.pt'
            if path_test_img_tensor.exists() and path_test_mask_tensor.exists():
                continue
            try:
                with tempfile.TemporaryDirectory() as tempfile_dir, tempfile.TemporaryDirectory() as quarantine_dir:
                    tempfile_dir = Path(tempfile_dir)

                    img_path = f'combinedsegmentationmodel/validationimages/{file_name}.jpg'
                    mask_path = f'combinedsegmentationmodel/validationlabels/{file_name}.png'
                    zip_ref.extract(img_path, tempfile_dir)
                    zip_ref.extract(mask_path, tempfile_dir)

                    quarantine_dir = Path(quarantine_dir)
                    process_image(source=tempfile_dir / img_path, dest=quarantine_dir / 'imgs' / file_name)
                    process_mask(source=tempfile_dir / mask_path, dest=quarantine_dir / 'masks' / file_name)

                    for fname in quarantine_dir.glob('**/imgs/*.*'):
                        shutil.move(src=fname, dst=dest_test_images_dir / fname.name)
                    for fname in quarantine_dir.glob('**/masks/*.*'):
                        shutil.move(src=fname, dst=dest_test_masks_dir / fname.name)
            except Exception as e:
                print(f'Error processing {file_name}. Error: {e}')

    dataset = SegFoodDataset(root=dest_dir / 'train')
    print(f"Train data shape: X.shape={dataset.X.shape}, y.shape={dataset.y.shape}")

    dataset = SegFoodDataset(root=dest_dir / 'test')
    print(f"Test data shape: X.shape={dataset.X.shape}, y.shape={dataset.y.shape}")


if __name__ == '__main__':
    main()
