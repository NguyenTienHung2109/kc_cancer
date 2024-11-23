import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import os
import io
import torch
from torchvision import transforms
import math
import xml.etree.ElementTree as ET

class Customdataset(Dataset):
    def __init__(
        self,
        dataset_path = "D:/ilib/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml",
        root_dir  = "D:/ilib/ibug_300W_large_face_landmark_dataset/",
        kpt_transform =None,
        img_transform = None,):
        super().__init__()
        self.dataset_path = dataset_path
        self.root_dir = root_dir
        self.img_transform = img_transform

        tree = ET.parse(dataset_path)
        root = tree.getroot()
        self.box = []
        self.img_size = []
        self.img_path = []
        each_img = root.findall('images/image')
        for i, img in enumerate(each_img):
            width = int(img.attrib['width'])
            height = int(img.attrib['height'])
            path = img.attrib['file']
            box_top = int(img.find('box').attrib['top'])
            box_left = int(img.find('box').attrib['left'])
            box_height = int(img.find('box').attrib['height'])
            box_width = int(img.find('box').attrib['width'])
            self.box.append([box_top, box_left, box_width, box_height])
            self.img_size.append([width, height])
            self.img_path.append([path])

        # Create empty array to hold points for all images
        num_images = len(root.findall('images/image'))
        self.points = np.zeros((num_images, 68, 2), dtype=np.float32)

        # Loop through images and extract points
        for i, image in enumerate(root.findall('images/image')):
            for j, part in enumerate(image.find('box').findall('part')):
                self.points[i, j, 0] = float(part.attrib['x'])
                self.points[i, j, 1] = float(part.attrib['y'])
    
        new_path = []
        for i in self.img_path:
            strip_i = str(i)
            new_path.append(self.root_dir + strip_i.strip("[],'"))
        self.new_path = []
        for i in new_path:
            self.new_path.append(i)
    
    def __len__(self):
        return len(self.img_path)
    
    def extractbbox(self, index):
        top = self.box[index][0]
        left = self.box[index][1]
        width =self.box[index][2]
        height = self.box[index][3]

        bottom = top + height
        right = left + width
        
        return top, left, bottom, right

    def __getitem__(self, index): 
        img = cv2.imread(self.new_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmark = self.points[index]
        
        top, left, bottom, right = self.extractbbox(index)
        if True:
            transform = A.Compose([
                A.Crop(left, top, right, bottom),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
            , keypoint_params= A.KeypointParams(format= 'xy', remove_invisible= False, angle_in_degrees = True )
            )
            
            transformed = transform(image = img, keypoints = landmark )
            img = transformed['image']
            landmark = transformed['keypoints']
            
        return img, landmark
    



