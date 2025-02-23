from typing import Any, Dict, Optional, Tuple, List

import torch
import numpy as np
import rootutils
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from albumentations import Compose

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.dataset.lidc_seg_dataset import LIDCSegDataset

class LIDCSegTransformDataset(Dataset):

    def __init__(self, dataset: Dataset, transform: Optional[Compose] = None):
            self.dataset = dataset

            assert transform is not None, ('transform is None')
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        transformed = self.transform(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]

        return image, mask

class LIDCSegDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        meta_file: str = None,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        train_val_test_meta_file : Tuple[str, str, str] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: int = 512,
        transform_train: Optional[Compose] = None,
        transform_val: Optional[Compose] = None,
        cache_data: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
    
    @property
    def image_size(self) -> int:
        return self.hparams.image_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.train_val_test_meta_file:
                train_meta_file, val_meta_file, test_meta_file = self.hparams.train_val_test_meta_file
                
                self.data_train = LIDCSegDataset(data_dir=self.hparams.data_dir, 
                                                  meta_file=train_meta_file, 
                                                  cache_data=self.hparams.cache_data)
                self.data_val = LIDCSegDataset(data_dir=self.hparams.data_dir, 
                                                meta_file=val_meta_file,
                                                cache_data=self.hparams.cache_data)
                self.data_test = LIDCSegDataset(data_dir=self.hparams.data_dir, 
                                                meta_file=test_meta_file,
                                                cache_data=self.hparams.cache_data)

            else:
                dataset = LIDCSegDataset(self.hparams.data_dir, 
                                          meta_file=self.hparams.meta_file,
                                          cache_data=self.hparams.cache_data)

                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(12345),
                )

            
            self.data_train = LIDCSegTransformDataset(
                dataset=self.data_train, transform=self.hparams.transform_train)
            self.data_val = LIDCSegTransformDataset(
                dataset=self.data_val, transform=self.hparams.transform_val)
            self.data_test = LIDCSegTransformDataset(
                dataset=self.data_test, transform=self.hparams.transform_val)
            
            print('Train-Val-Test:', len(self.data_train), len(self.data_val),
                  len(self.data_test))
            
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "data")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="lidc_seg.yaml")
    def main(cfg: DictConfig):
        h, w = 352, 352
        cfg["transform_train"]["transforms"][0].height = h
        cfg["transform_train"]["transforms"][0].width = w
        cfg["transform_val"]["transforms"][0].height = h
        cfg["transform_val"]["transforms"][0].width = w

        print(cfg)
        datamodule: LIDCSegDataModule = hydra.utils.instantiate(cfg, data_dir=f"{root}/data")
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        print("Length of train dataloader:", len(train_dataloader))

        batch_image = next(iter(train_dataloader))
        images, masks = batch_image
        
        print("image:", images.shape, images.dtype)
        print("mask:", masks.shape, masks.dtype)

        n_row = 3
        import matplotlib.pyplot as plt
        import cv2
        fig, axs = plt.subplots(n_row, 2, figsize=(15, 10))

        for i in range(n_row):
            image = images[i, 0]
            mask = masks[i, 0]

            image = torch.stack([image, image, image], dim=-1).numpy()
            mask = torch.stack([mask, mask, mask], dim=-1).numpy()

            axs[i][0].imshow(image)
            axs[i][0].set_title('Image')
            axs[i][0].axis("off")

            axs[i][1].imshow(mask)
            axs[i][1].set_title('Mask')
            axs[i][1].axis("off")
        
        plt.show()

    main()