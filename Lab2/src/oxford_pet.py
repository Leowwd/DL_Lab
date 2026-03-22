import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, txt_name, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        
        txt_path = os.path.join(root_dir, txt_name)
        with open(txt_path, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines() if line.strip()]
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, f"{filename}.jpg")
        target_size = (256, 256)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        
        if self.mode == "test": return image_tensor
        
        mask_path = os.path.join(self.mask_dir, f"{filename}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)        
        mask = np.where(mask > 1, 0, 1).astype(np.float32)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)   
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return image_tensor, mask_tensor
        