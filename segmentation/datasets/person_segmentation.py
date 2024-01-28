import typing as t
import zipfile
from pathlib import Path

import kaggle
import PIL.Image as PIL_Image
import torch as T
import torchvision.transforms as TV_trfs
from torch.utils.data import Dataset
from tqdm import tqdm, trange


def _download(dataset_name: str, path: Path) -> None:
    api = kaggle.api
    api.authenticate()
    path.parent.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(
        dataset=dataset_name, path=path, unzip=False, quiet=False)


def preprocess(img_path: Path, is_mask: bool) -> None:
    transformations = TV_trfs.Compose([
        TV_trfs.Resize(
            (256, 256),
            antialias=True,  # type: ignore
        ),
        TV_trfs.ToTensor(),
    ])

    with PIL_Image.open(img_path) as img:
        # If image if not grayscale, convert it to RGB.
        if is_mask:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        tensor = transformations(img)
        T.save(tensor, img_path.with_suffix('.pt'))


class PersonSegmentationDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        masks_dir: Path,
    ):
        self.img_dir = img_dir
        self.masks_dir = masks_dir

        self.img_paths = sorted(list(self.img_dir.glob('*.pt')))
        self.masks_paths = sorted(list(self.masks_dir.glob('*.pt')))

        self.imgs = T.empty((len(self.img_paths), 3, 256, 256))
        self.masks = T.empty((len(self.masks_paths), 1, 256, 256))

        for i in trange(len(self.img_paths)):
            img_path = self.img_paths[i]
            mask_path = self.masks_paths[i]
            assert img_path.stem == mask_path.stem
            self.imgs[i] = T.load(img_path)
            self.masks[i] = T.load(mask_path)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> t.Tuple[T.Tensor, T.Tensor]:
        return self.imgs[idx], self.masks[idx]


def main(
    dest_dir: Path = Path('./data/'),
    temp_dir: Path = Path('./temp/'),
    label: str = 'person_segmentation',
):
    temp_dir = Path(temp_dir) / label
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)
        _download(
            dataset_name='tapakah68/supervisely-filtered-segmentation-person-dataset',
            path=temp_dir,
        )
    zip_path = next(temp_dir.glob('*.zip'))

    dest_dir = dest_dir / label
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Iterate over the file names
            file_list = zip_ref.namelist()
            for file_name in tqdm(file_list, desc='Extracting files...'):
                file_path = Path(file_name)

                if file_path.parent.name == 'collage':
                    continue

                suffix = file_path.suffix.lower()
                match suffix:
                    case '.png':
                        zip_ref.extract(file_name, temp_dir)
                        file_src = temp_dir / file_name
                        file_dst = dest_dir / file_src.parent.name / file_src.name
                        file_dst.parent.mkdir(parents=True, exist_ok=True)
                        file_src.rename(file_dst)

                        if file_dst.parent.name == 'images':
                            preprocess(file_dst, is_mask=False)
                        elif file_dst.parent.name == 'masks':
                            preprocess(file_dst, is_mask=True)
                        else:
                            raise ValueError(
                                f'Unexpected image parent folder: {file_dst}'
                            )

                    case '.csv':
                        zip_ref.extract(file_name, dest_dir)
                    case _:
                        raise ValueError(
                            f'Unexpected file extension: {suffix}'
                        )

    assert dest_dir.exists()
    assert (dest_dir / 'images').exists()
    assert (dest_dir / 'masks').exists()
    assert (dest_dir / 'df.csv').exists()

    psd = PersonSegmentationDataset(
        img_dir=dest_dir / 'images',
        masks_dir=dest_dir / 'masks',
    )

    x, y = psd[0]
    assert x.shape == (3, 256, 256)
    assert y.shape == (1, 256, 256)


if __name__ == '__main__':
    main()
