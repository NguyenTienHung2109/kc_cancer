import cv2
import os
# import random 
import numpy as np
from PIL import Image
# import imutils
import matplotlib.pyplot as plt
from math import *
import random
import xml.etree.ElementTree as ET 
from torch.utils.data import Dataset
# import torch
import torchvision.transforms.functional as TF
# from torchvision import datasets, models, transforms

class FaceLandmarksDataset(Dataset):
    def __init__(self, img_paths, keypoints, transform=None):
        self.img_paths = img_paths
        self.keypoints = keypoints
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        keypoints = self.keypoints[index]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            # Apply the same transformation to the image and keypoints
            transform_output = self.transform(image=np.array(img), keypoints=keypoints)
            img = transform_output['image']
            keypoints = transform_output['keypoints']
            
        return img, keypoints