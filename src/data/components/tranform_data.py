# from dlib300W_extract_data import FaceLandmarksDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import math
import cv2
import os
import random 
import numpy as np
from PIL import Image
import imutils
import matplotlib.pyplot as plt
from math import *
import random
import xml.etree.ElementTree as ET 
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset

class Transforms():
    def __init__(self):
        pass

    def crop_face(self, image, landmarks, crops):
        top = int(crops[0])
        left = int(crops[1])
        width = int(crops[2])
        height = int(crops[3])
        
        image = TF.crop(image, top, left, height, width)
        img_shape = np.array(image).shape

        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks/torch.tensor([img_shape[1], img_shape[0]])

        return image, landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)

        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness = random.random(),
                    contrast = random.random(),
                    saturation= random.random(),
                    hue = random.uniform(0,0.5))
        image =  color_jitter(image)
        return image, landmarks

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))],
            
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks -0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks +0.5
        return  Image.fromarray(image), new_landmarks

    def __call__ (self, image, landmarks, crops):
        image = Image.fromarray(image)
        
        # aumentation
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle = random.randint(-20, 20))

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])

        return image, landmarks
    
    