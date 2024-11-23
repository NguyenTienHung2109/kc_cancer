from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.transforms import transforms
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
import math
import cv2
import os
import random 
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from math import *
import random
import xml.etree.ElementTree as ET 
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import sys
sys.path.append("D:/ilib/hydra/lightning-hydra-template/")

#from src.data.components.extract_data import FaceLandmarksDataset
#from src.data.components.transform_data import Transforms
from src.data.components.Customdataset import Customdataset

class DLIB300WDataModule(LightningDataModule):
    def __init__(
        self,
        data_path = 'D:/ilib/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml',
        root_dir  = 'D:/ilib/ibug_300W_large_face_landmark_dataset/',
        train_val_test_split: Tuple[int, int, int] = (5000, 1000, 666),
        batch_size = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        
        super().__init__()
        self.save_hyperparameters(logger = False)
        
        self.data_path = data_path
        self.root_dir =  root_dir
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
    def prepare_data(self):
        pass
        
        
    def setup(self, stage: Optional[str] = None):
        
        dataset = Customdataset(self.data_path, self.root_dir)
        
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset= dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            
        
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def draw_batch(self):
        images, landmarks = next(iter(self.train_dataloader()))
        batch_size = len(images)
        grid_size = math.sqrt(batch_size)
        grid_size = math.ceil(grid_size)
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
        for i in range(batch_size):
            ax = fig.add_subplot(grid_size, grid_size, i+1, xticks=[], yticks=[])
            image, landmark = images[i], landmarks[i]
            image = image.squeeze().permute(1,2,0)
            plt.imshow(image)
            kpt = []
            for j in range(68):
                kpt.append(plt.plot(landmark[j][0], landmark[j][1], 'g.'))
        plt.tight_layout()
        plt.show()
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = DLIB300WDataModule()
