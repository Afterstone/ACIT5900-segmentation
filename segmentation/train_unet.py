import datetime as dt
import typing as t
from calendar import c
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.logger import Logger
from matplotlib import axis
from matplotlib.pylab import f
from pandas import isna
from torch.utils.data import DataLoader

from segmentation.datasets.seg_food import SegFoodDataset


class Augmentation(T.nn.Module):
    ...


class AugmentationList(T.nn.Module):
    def __init__(self, augmentations: t.List[Augmentation]):
        super().__init__()
        self.augmentations = augmentations

    def forward(self, img, mask):
        for aug in self.augmentations:
            img, mask = aug(img, mask)
        return img, mask


class RandomVerticalFlip(Augmentation):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        probs = T.rand(1)
        if probs[0] < self.p:
            img = TVTF.vflip(img)
            mask = TVTF.vflip(mask)
        return img, mask


class RandomHorizontalFlip(Augmentation):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        probs = T.rand(1)
        if probs[0] < self.p:
            img = TVTF.hflip(img)
            mask = TVTF.hflip(mask)
        return img, mask


class RandomPerspective(Augmentation):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        fill_values = [0.0]
        startpoints = [[0, 0], [0, 256], [256, 256], [256, 0]]

        offsets = T.randint(-64, 64, (4, 2))
        endpoints = [
            [0 + offsets[0, 0], 0 + offsets[0, 1]],
            [0 + offsets[1, 0], 256 + offsets[1, 1]],
            [256 + offsets[2, 0], 256 + offsets[2, 1]],
            [256 + offsets[3, 0], 0 + offsets[3, 1]],
        ]
        endpoints = np.clip(endpoints, -100, 256+100).tolist()
        img = TVTF.perspective(img, startpoints, endpoints, fill=fill_values)
        mask = TVTF.perspective(mask, startpoints, endpoints, fill=fill_values)
        return img, mask


class RandomNoise(Augmentation):
    def __init__(self, std: float, mean: float = 0.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img, mask):
        noise = T.randn_like(img) * self.std + self.mean
        img = img + noise
        return img, mask


class Unet(L.LightningModule):
    def __init__(
        self,
        test_data_X: T.Tensor,
        test_data_y: T.Tensor,
        lr: float = 1e-3,
        test_thresh: float = 0.5,
        mask_norm_constant: float = 0.5,
        augmentations: AugmentationList | None = None,
        log_dir: Path = Path('.'),
    ):
        super().__init__()

        self.log_dir = log_dir / 'images'
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        if test_data_X.shape[0] > 10:
            raise ValueError('test_data must be a tensor of shape (N, 3, H, W) where N < 10')
        self.test_data_X = test_data_X.detach()
        self.test_data_y = test_data_y.detach()
        self.test_thresh = test_thresh
        self.augmentations = augmentations if augmentations is not None else AugmentationList([])

        self.mask_norm_constant = mask_norm_constant

        self.lr = lr

        self.relu = T.nn.ReLU()
        self.sigmoid = T.nn.Sigmoid()
        self.maxpool = T.nn.MaxPool2d(2)

        self.bn1 = T.nn.BatchNorm2d(3)
        self.enc1 = T.nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = T.nn.Conv2d(32, 64, 3, padding=1)
        self.enc3 = T.nn.Conv2d(64, 128, 3, padding=1)
        self.enc4 = T.nn.Conv2d(128, 256, 3, padding=1)

        self.ebn1 = T.nn.BatchNorm2d(32)
        self.ebn2 = T.nn.BatchNorm2d(64)
        self.ebn3 = T.nn.BatchNorm2d(128)
        self.ebn4 = T.nn.BatchNorm2d(256)

        self.dec4 = T.nn.Conv2d(256, 128, 3, padding=1)
        self.dec3 = T.nn.Conv2d(256, 64, 3, padding=1)
        self.dec2 = T.nn.Conv2d(128, 32, 3, padding=1)
        self.dec1 = T.nn.Conv2d(64, 1, 3, padding=1)

        self.dbn4 = T.nn.BatchNorm2d(128)
        self.dbn3 = T.nn.BatchNorm2d(64)
        self.dbn2 = T.nn.BatchNorm2d(32)
        self.dbn1 = T.nn.BatchNorm2d(1)

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, X: T.Tensor) -> T.Tensor:
        X = self.bn1(X)

        E1 = self.relu(self.ebn1(self.enc1(X)))
        E2 = self.relu(self.ebn2(self.enc2(self.maxpool(E1))))
        E3 = self.relu(self.ebn3(self.enc3(self.maxpool(E2))))
        E4 = self.relu(self.ebn4(self.enc4(self.maxpool(E3))))

        D4 = self.relu(self.dbn4(self.dec4(E4)))
        D4 = T.nn.functional.interpolate(D4, scale_factor=2, mode='bilinear', align_corners=True)
        D3 = self.relu(self.dbn3(self.dec3(T.concat([E3.real, D4], dim=1))))
        D3 = T.nn.functional.interpolate(D3, scale_factor=2, mode='bilinear', align_corners=True)
        D2 = self.relu(self.dbn2(self.dec2(T.concat([E2.real, D3], dim=1))))
        D2 = T.nn.functional.interpolate(D2, scale_factor=2, mode='bilinear', align_corners=True)
        D1 = self.dbn1(self.dec1(T.concat([E1, D2], dim=1)))
        return D1

    def predict(self, X: T.Tensor) -> T.Tensor:
        y_hat = self.forward(X)
        y_hat = self.sigmoid(y_hat)
        return y_hat

    def predict_augmented(self, X: T.Tensor) -> T.Tensor:
        X_repeated = X.repeat(4, 1, 1, 1)
        X_repeated[1] = TVTF.vflip(X_repeated[1])
        X_repeated[2] = TVTF.hflip(X_repeated[2])
        X_repeated[3] = TVTF.vflip(TVTF.hflip(X_repeated[3]))
        y_hat = self.forward(X_repeated)
        y_hat[1] = TVTF.vflip(y_hat[1])
        y_hat[2] = TVTF.hflip(y_hat[2])
        y_hat[3] = TVTF.vflip(TVTF.hflip(y_hat[3]))
        y_hat = self.sigmoid(y_hat)
        y_hat = y_hat.mean(dim=0)
        return y_hat

    def training_step(self, batch: t.Tuple[T.Tensor, T.Tensor], _: int) -> T.Tensor:
        X, y = self.augmentations(*batch)

        y_hat = self.forward(X)
        loss = T.nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=T.tensor(self.mask_norm_constant))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        with T.no_grad():
            y_class = self.sigmoid(y_hat)
            y_class[y_class > self.test_thresh] = 1.0
            y_class[y_class <= self.test_thresh] = 0.0
            y_class = y_class.long()
            y_long = y.long()

            acc = T.sum(y_class == y_long) / T.numel(y_long)
            self.log('train_acc', acc, on_epoch=True,  on_step=False, logger=True)

            # Shortcut: Since we are using binary labels, we can simply sum over the entire tensor
            # to find the number of true positives, false positives, and false negatives.
            # We can use y_hat * y to find the true positives, since we've thresholded y_hat.
            TP = T.sum(y_class * y).item()
            FP = T.sum(y_class * (1 - y)).item()
            FN = T.sum((1 - y_class) * y).item()

            # TP / (TP + FP), i.e. where y_hat == y divided by predicted positives.
            # Check if sum of y_class == 0 or if it contains nans.
            precision = 0.0 if int(TP) == 0 else TP / (TP + FP)
            self.log('train_precision', precision, on_epoch=True,  on_step=False, logger=True)

            # TP / (TP + FN), i.e. where y_hat == y divided by actual positives.
            recall = 0.0 if int(TP) == 0 else TP / (TP + FN)
            self.log('train_recall', recall, on_epoch=True,  on_step=False, logger=True)

            # 2 * (precision * recall) / (precision + recall)
            if int(FP + FN) == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)
            self.log('train_f1_score', f1_score, on_epoch=True,  on_step=False, logger=True)

            # IoU = TP / (TP + FP + FN)
            iou = (TP / (T.sum(y_class + y_long) - TP)).item()
            self.log('train_iou', iou, on_epoch=True,  on_step=False, logger=True)

        return loss

    # Run after training epoch.
    def on_train_epoch_end(self) -> None:
        with T.no_grad():
            test_data_X = self.test_data_X.to(self.device)
            test_data_y = self.test_data_y
            y = self.predict(test_data_X).detach().cpu().numpy()

            y_thresh = y.copy()
            y_thresh[y > self.test_thresh] = 1.0
            y_thresh[y <= self.test_thresh] = 0.0

            y_augmented = T.empty(y.shape, dtype=T.float32)
            for i in range(test_data_X.shape[0]):
                y_augmented[i] = self.predict_augmented(test_data_X[i].unsqueeze(0)).detach().cpu()
            y_augmented = y_augmented.numpy()
            y_aug_thresh = y_augmented.copy()
            y_aug_thresh[y_augmented > self.test_thresh] = 1.0
            y_aug_thresh[y_augmented <= self.test_thresh] = 0.0

            test_data_X = test_data_X.cpu()
            n_imgs = test_data_X.shape[0]
            fig, axs = plt.subplots(n_imgs, 6, figsize=(20, 2 * n_imgs))
            for i in range(n_imgs):
                axs[i, 0].imshow(test_data_X[i].permute(1, 2, 0))
                axs[i, 1].imshow(test_data_y[i].squeeze(), vmin=0, vmax=1)
                axs[i, 2].imshow(y[i].squeeze(), vmin=0, vmax=1)
                axs[i, 3].imshow(y_thresh[i].squeeze(), vmin=0, vmax=1)
                axs[i, 4].imshow(y_augmented[i].squeeze(), vmin=0, vmax=1)
                axs[i, 5].imshow(y_aug_thresh[i].squeeze(), vmin=0, vmax=1)

            for ax in axs.flat:
                ax.set_xticks([])
                ax.set_yticks([])

            # Get current epoch
            epoch = self.current_epoch
            fig.savefig(self.log_dir / f'./test_{epoch}.png')

    def validation_step(self, batch: t.Tuple[T.Tensor, T.Tensor], _: int) -> None:
        X, y = batch
        y_hat = self.forward(X.detach())

        loss = T.nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=T.tensor(self.mask_norm_constant))
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)

        y_class = self.sigmoid(y_hat)
        y_class[y_class > self.test_thresh] = 1.0
        y_class[y_class <= self.test_thresh] = 0.0
        y_class = y_class.long()
        y_long = y.long()

        acc = T.sum(y_class == y_long) / T.numel(y_long)
        self.log('val_acc', acc, on_epoch=True,  on_step=False, logger=True)

        # Shortcut: Since we are using binary labels, we can simply sum over the entire tensor
        # to find the number of true positives, false positives, and false negatives.
        # We can use y_hat * y to find the true positives, since we've thresholded y_hat.
        TP = T.sum(y_class * y).item()
        FP = T.sum(y_class * (1 - y)).item()
        FN = T.sum((1 - y_class) * y).item()

        # TP / (TP + FP), i.e. where y_hat == y divided by predicted positives.
        # Check if sum of y_class == 0 or if it contains nans.
        precision = 0.0 if int(TP) == 0 else TP / (TP + FP)
        self.log('val_precision', precision, on_epoch=True,  on_step=False, logger=True)

        # TP / (TP + FN), i.e. where y_hat == y divided by actual positives.
        recall = 0.0 if int(TP) == 0 else TP / (TP + FN)
        self.log('val_recall', recall, on_epoch=True,  on_step=False, logger=True)

        # 2 * (precision * recall) / (precision + recall)
        if int(TP) == 0 or int(precision + recall) < 1e-10:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        self.log('val_f1_score', f1_score, on_epoch=True,  on_step=False, logger=True)

        # IoU = TP / (TP + FP + FN)
        iou = (TP / (T.sum(y_class + y_long) - TP)).item()
        self.log('val_iou', iou, on_epoch=True,  on_step=False, logger=True)


def main(
    log_dir: Path = Path('./logs/unet/'),
    debug: bool = False,
):
    log_name = 'unet_debug' if debug else 'unet'
    log_name = f"{log_name}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if debug:
        ds_train = SegFoodDataset(root=Path('./data/seg_food/test'), load_first_n=100)
        ds_test = ds_train
    else:
        ds_train = SegFoodDataset(root=Path('./data/seg_food/train'))
        ds_test = SegFoodDataset(root=Path('./data/seg_food/test'))
    dl_train = DataLoader(dataset=ds_train, batch_size=32, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=4, shuffle=False)

    augmentations = AugmentationList([
        RandomVerticalFlip(p=0.5),
        RandomHorizontalFlip(p=0.5),
        RandomPerspective(p=0.5),
        RandomNoise(std=0.3),
    ])

    _, ax = plt.subplots(4, 4)
    X = ds_train.X[:4].detach()
    y = ds_train.y[:4].detach()

    for i, (x_i, y_i) in enumerate(zip(X, y)):
        x_i = x_i.unsqueeze(0)
        y_i = y_i.unsqueeze(0)
        x_aug, y_aug = augmentations(x_i, y_i)
        ax[i, 0].imshow(x_i.squeeze().permute(1, 2, 0))
        ax[i, 1].imshow(y_i.squeeze(), vmin=0, vmax=1)
        ax[i, 2].imshow(x_aug.squeeze().permute(1, 2, 0))
        ax[i, 3].imshow(y_aug.squeeze(), vmin=0, vmax=1)

    plt.savefig('./augmentation.png')

    model = Unet(
        test_data_X=ds_test.X[:10],
        test_data_y=ds_test.y[:10],
        mask_norm_constant=1 - float(ds_train.y.mean().item()),
        augmentations=augmentations,
        log_dir=log_dir / log_name,
        lr=1e-3,
    )

    loggers: list[Logger] = [TensorBoardLogger(log_dir, name=log_name)]
    callbacks: list[Callback | Checkpoint] = [
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
    ]

    trainer = L.Trainer(max_epochs=50, accelerator='auto', logger=loggers, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_test)


if __name__ == '__main__':
    main()
