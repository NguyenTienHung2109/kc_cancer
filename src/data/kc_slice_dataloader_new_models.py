from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import rootutils
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from albumentations import Compose

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.dataset.kc_cancer_new_models import KCSliceDataset

colors = {
    "nodule": (1., 1., 0.),
    "bbox": (0., 1., 0.),
    "lung_loc": [(0.5, 0.75, 1.0), (0.0, 0.5, 0.75), 
                (0.0, 0.0, 1.0), (1.0, 0.75, 0.5), (1.0, 0.5, 0.0)]
}

def draw_masks(slice ,masks, colors):
    # slice: (w, h)
    # masks: (w, h, c)
    # colors: tuple()

    colored_mask = np.zeros((*masks.shape[:2], 3), dtype=np.float32)
    masks = masks[:, :, 1:] # first dim is background
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        if mask.sum() > 0:
            colored_mask += np.expand_dims(mask, axis=-1) * colors[i]
            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(slice, contour, -1, colors[i], 1)
    return slice, colored_mask

class TransformDataset(Dataset):

    def __init__(self, 
                dataset: Dataset, 
                use_nodule_cls: bool = False,
                use_lung_pos: bool = False,
                use_lung_loc: bool = False,
                use_lung_damage_cls: bool = False,
                transform: Optional[Compose] = None):
        self.dataset = dataset
        self.use_nodule_cls = use_nodule_cls
        self.use_lung_pos = use_lung_pos
        self.use_lung_loc = use_lung_loc
        self.use_lung_damage_cls = use_lung_damage_cls
        
        assert transform is not None, ('transform is None')
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datapoint = self.dataset[idx]
        # debug = True
        # if debug:
        #     print(datapoint["slice_info"])
        #     print(len(datapoint["nodule_info"]["bbox"]))
        #     for key in datapoint["nodule_info"]["label"].keys():
        #         print(key, datapoint["nodule_info"]["label"][key].shape)
        #     print('-' * 20)

        bboxes = None
        if self.use_nodule_cls or self.use_lung_damage_cls:
            bboxes = datapoint["nodule_info"]["bbox"]
        
        lung_loc_mask = None
        if self.use_lung_loc:
            lung_loc_mask = datapoint["seg_lung_loc"]

        if bboxes is not None:
            fake_label = [i for i in range(bboxes.shape[0])]
            if lung_loc_mask is not None:
                transformed = self.transform(image=datapoint["slice"], 
                                            mask=datapoint["seg_nodule"],
                                            lung_loc=lung_loc_mask,
                                            bboxes=bboxes, 
                                            class_labels=fake_label)
            else:
                transformed = self.transform(image=datapoint["slice"], 
                                            mask=datapoint["seg_nodule"], 
                                            bboxes=bboxes, 
                                            class_labels=fake_label)
        else:
            if lung_loc_mask is not None:
                transformed = self.transform(image=datapoint["slice"],
                                            mask=datapoint["seg_nodule"],
                                            lung_loc=lung_loc_mask)
            else:
                transformed = self.transform(image=datapoint["slice"],
                                            mask=datapoint["seg_nodule"])

        datapoint["slice"] = transformed["image"]
        datapoint["seg_nodule"] = transformed["mask"].unsqueeze(0)
        
        if self.use_nodule_cls or self.use_lung_damage_cls:
            datapoint["nodule_info"]["bbox"] = transformed["bboxes"].astype(np.int64)
            # get label, To-Tensor
            for key in datapoint["nodule_info"]["label"]:
                datapoint["nodule_info"]["label"][key] = torch.tensor(datapoint["nodule_info"]["label"][key])

        if self.use_lung_loc:
            lung_loc_mask = torch.argmax(transformed["lung_loc"], dim=0)
            lung_loc_mask = F.one_hot(lung_loc_mask, num_classes=transformed["lung_loc"].shape[0])
            datapoint["seg_lung_loc"] = lung_loc_mask.permute(2, 0, 1).to(torch.float32)

        # if debug:
        #     print("After transform")
        #     print(len(datapoint["nodule_info"]["bbox"]))
        #     for key in datapoint["nodule_info"]["label"].keys():
        #         print(key, datapoint["nodule_info"]["label"][key].shape)
        #     print('-' * 20)

        return datapoint

class KCSliceDataModule(LightningDataModule):
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
        use_nodule_cls: bool = False,
        use_lung_pos: bool = False,
        use_lung_loc: bool = False,
        use_lung_damage_cls: bool = False,
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

    def get_subset(self, dataset: Dataset, n_dataset: int):
        # get subset of dataset to test code before training

        if 1 < n_dataset and n_dataset < len(dataset):
            print(len(dataset), "-->", n_dataset)
            return Subset(dataset, list(range(n_dataset)))

        return dataset

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
                train_set = KCSliceDataset(data_dir=self.hparams.data_dir, 
                                            meta_file=train_meta_file,
                                            use_nodule_cls=self.hparams.use_nodule_cls,
                                            use_lung_pos=self.hparams.use_lung_pos,
                                            use_lung_loc=self.hparams.use_lung_loc,
                                            use_lung_damage_cls=self.hparams.use_lung_damage_cls,
                                            cache_data=self.hparams.cache_data)
                val_set = KCSliceDataset(data_dir=self.hparams.data_dir,
                                        meta_file=val_meta_file,
                                        use_nodule_cls=self.hparams.use_nodule_cls,
                                        use_lung_pos=self.hparams.use_lung_pos,
                                        use_lung_loc=self.hparams.use_lung_loc,
                                        use_lung_damage_cls=self.hparams.use_lung_damage_cls,
                                        cache_data=self.hparams.cache_data)
                test_set = KCSliceDataset(data_dir=self.hparams.data_dir,
                                        meta_file=test_meta_file,
                                        use_nodule_cls=self.hparams.use_nodule_cls,
                                        use_lung_pos=self.hparams.use_lung_pos,
                                        use_lung_loc=self.hparams.use_lung_loc,
                                        use_lung_damage_cls=self.hparams.use_lung_damage_cls,
                                        cache_data=self.hparams.cache_data)

                # train_set = self.get_subset(train_set, 100)
                # val_set = self.get_subset(val_set, 10)
                # test_set = self.get_subset(test_set, 10)

            else:
                dataset = KCSliceDataset(self.hparams.data_dir, 
                                        meta_file=self.hparams.meta_file,
                                        use_nodule_cls=self.hparams.use_nodule_cls,
                                        use_lung_pos=self.hparams.use_lung_pos,
                                        use_lung_loc=self.hparams.use_lung_loc,
                                        cache_data=self.hparams.cache_data)

                train_set, val_set, test_set = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(12345),
                )
            
            self.data_train = TransformDataset(dataset=train_set, 
                                            use_nodule_cls=self.hparams.use_nodule_cls,
                                            use_lung_pos=self.hparams.use_lung_pos,
                                            use_lung_loc=self.hparams.use_lung_loc,
                                            use_lung_damage_cls=self.hparams.use_lung_damage_cls,
                                            transform=self.hparams.transform_train)
            self.data_val = TransformDataset(dataset=val_set, 
                                            use_nodule_cls=self.hparams.use_nodule_cls,
                                            use_lung_pos=self.hparams.use_lung_pos,
                                            use_lung_loc=self.hparams.use_lung_loc,
                                            use_lung_damage_cls=self.hparams.use_lung_damage_cls,
                                            transform=self.hparams.transform_val)
            self.data_test = TransformDataset(dataset=test_set, 
                                            use_nodule_cls=self.hparams.use_nodule_cls,
                                            use_lung_pos=self.hparams.use_lung_pos,
                                            use_lung_loc=self.hparams.use_lung_loc,
                                            use_lung_damage_cls=self.hparams.use_lung_damage_cls,
                                            transform=self.hparams.transform_val)

            print('Train-Val-Test:', len(self.data_train), len(self.data_val), len(self.data_test))

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
                config_name="kc_slice_new_models.yaml")
    def main(cfg: DictConfig):
        h, w = 128, 128
        version = 4.3
        cfg["train_val_test_meta_file"] = ["train_meta_info.csv", 
                                            "val_meta_info.csv",
                                            "test_meta_info.csv"]
        cfg["data_dir"] = f"{root}/data/kc_cancer_{version}"
        cfg["use_nodule_cls"] = True
        cfg["use_lung_pos"] = True
        cfg["use_lung_loc"] = True
        cfg["use_lung_damage_cls"] = True
        
        cfg["transform_train"]["transforms"][0].height = h
        cfg["transform_train"]["transforms"][0].width = w
        cfg["transform_val"]["transforms"][0] = cfg["transform_train"]["transforms"][0]
        cfg["transform_val"]["bbox_params"] = cfg["transform_train"]["bbox_params"]
        cfg["transform_val"]["additional_targets"] = cfg["transform_train"]["additional_targets"]

        print(cfg)
        datamodule: KCSliceDataModule = hydra.utils.instantiate(cfg, num_workers=0)
        datamodule.setup()

        print(f"Data version {version}")
        train_dataloader = datamodule.train_dataloader()
        print("Length of train dataloader:", len(train_dataloader))
        
        for i, batch in enumerate(train_dataloader):
            print("~" * 50)
            print("batch:", i)
            break

        slices, slice_info = batch["slice"], batch["slice_info"]
        nodule_masks = batch["seg_nodule"]
        print("slice:", slices.shape, slices.dtype)
        print("nodule mask:", nodule_masks.shape, nodule_masks.dtype)

        if cfg["use_nodule_cls"] or cfg["use_lung_damage_cls"]:
            nodule_infos = batch["nodule_info"]
            print("bbox:", nodule_infos["bbox"].shape, nodule_infos["bbox"].dtype)
            for key in nodule_infos["label"].keys():
                print(f"{key}:", nodule_infos["label"][key].shape, nodule_infos["label"][key].dtype)

        if cfg["use_lung_pos"]:
            lung_pos_labels = batch["cls_lung_pos"]
            print("right_lung:", lung_pos_labels["right_lung"].shape, lung_pos_labels["right_lung"].dtype)
            print("left_lung:", lung_pos_labels["left_lung"].shape, lung_pos_labels["left_lung"].dtype)

        if cfg["use_lung_loc"]:
            lung_loc_masks = batch["seg_lung_loc"]
            print("lung_loc mask", lung_loc_masks.shape, lung_loc_masks.dtype)

        col_titles = ["Slices", "Nodule Mask"]
        if cfg["use_lung_loc"]:
            col_titles.append("Lung Location Mask")
        
        n_row , n_col= 3, len(col_titles)
        _, axes = plt.subplots(n_row, n_col, figsize=(15, 10))
        for i, title in enumerate(col_titles):
            axes[0, i].set_title(title, weight='bold')

        for i in range(n_row):
            slice = slices[i, 0]
            slice = torch.stack([slice, slice, slice], dim=-1).numpy()
            
            nodule_mask = nodule_masks[i][0]
            if nodule_mask.sum() > 0:
                nodule_contours, _ = cv2.findContours(np.array(nodule_mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(slice, nodule_contours, -1, colors["nodule"], 1)
            
            nodule_mask = torch.stack([nodule_mask, nodule_mask, nodule_mask], dim=-1).numpy()
            if cfg["use_nodule_cls"] or cfg["use_lung_damage_cls"]:
                bboxes = nodule_infos["bbox"][i]
                for bbox in bboxes:
                    bbox = [int(x) for x in bbox]
                    cv2.rectangle(slice, (bbox[0], bbox[1]),
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                    colors["bbox"], 1)
                    cv2.rectangle(nodule_mask, (bbox[0], bbox[1]),
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                    colors["bbox"], 1)

            axes[i][1].imshow(nodule_mask, cmap="gray")
            axes[i][1].axis("off")
            
            if cfg["use_lung_loc"]:
                lung_loc_mask = np.array(lung_loc_masks[i], dtype=np.uint8).transpose(1, 2, 0)
                slice, colored_mask = draw_masks(slice, masks=lung_loc_mask, colors=colors["lung_loc"])
                axes[i][2].imshow(colored_mask)
                axes[i][2].axis("off")
            
            axes[i, 0].text(-0.75, 0.5, slice_info[i], weight='bold', transform=axes[i, 0].transAxes)
            axes[i][0].imshow(slice)
            axes[i][0].axis("off")
        
        # plt.show()
        plt.savefig("dataloader.png")

    main()