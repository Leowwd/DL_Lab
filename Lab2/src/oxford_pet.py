import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, txt_name, mode="train", transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.target_size = (256, 256)
        
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "annotations", "trimaps")
        
        txt_path = os.path.join(root_dir, txt_name)
        with open(txt_path, "r") as f:
            self.filenames = [line.strip() for line in f.readlines() if line.strip()]
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):        
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, f"{filename}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == "test": 
            if self.transform:
                augmented = self.transform(image=image_tensor)
                image_tensor = augmented["image"]
            else:
                image = cv2.resize(image, self.target_size)
                
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            return torch.from_numpy(image)
        
        mask_path = os.path.join(self.mask_dir, f"{filename}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)        
        mask = np.where(mask > 1, 0, 1).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(image, self.target_size)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0)
        