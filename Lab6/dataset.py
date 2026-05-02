"""
dataset.py — iCLEVR Dataset for Conditional DDPM

Provides:
  - ICLEVRDataset: training dataset (image + multi-label one-hot condition)
  - get_test_conditions: load test.json / new_test.json as one-hot tensors
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ICLEVRDataset(Dataset):
    """Training dataset for iCLEVR.

    Each sample returns (image_tensor, one_hot_label) where:
      - image_tensor: [3, 64, 64], normalized to [-1, 1]
      - one_hot_label: [24], multi-hot binary vector
    """

    def __init__(self, img_dir, train_json_path, objects_json_path, img_size=64):
        """
        Args:
            img_dir: path to the folder containing training .png files
            train_json_path: path to train.json
            objects_json_path: path to objects.json
            img_size: target resolution (default 64)
        """
        super().__init__()
        self.img_dir = img_dir

        # Load label mapping: object_name -> index (0-23)
        with open(objects_json_path, 'r') as f:
            self.object_to_idx = json.load(f)
        self.num_classes = len(self.object_to_idx)  # 24

        # Load training annotations: filename -> [object_name, ...]
        with open(train_json_path, 'r') as f:
            self.train_data = json.load(f)

        # Build list of (filename, one_hot_label)
        self.samples = []
        for filename, objects in self.train_data.items():
            one_hot = self._objects_to_onehot(objects)
            self.samples.append((filename, one_hot))

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),                               # [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ])

    def _objects_to_onehot(self, objects):
        """Convert a list of object names to a 24-dim one-hot vector."""
        one_hot = torch.zeros(self.num_classes)
        for obj in objects:
            if obj in self.object_to_idx:
                one_hot[self.object_to_idx[obj]] = 1.0
        return one_hot

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, one_hot = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, one_hot


def get_test_conditions(test_json_path, objects_json_path):
    """Load test conditions as a batch of one-hot tensors.

    Args:
        test_json_path: path to test.json or new_test.json
        objects_json_path: path to objects.json

    Returns:
        conditions: Tensor of shape [N, 24] with multi-hot labels
    """
    with open(objects_json_path, 'r') as f:
        object_to_idx = json.load(f)
    num_classes = len(object_to_idx)

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    conditions = []
    for objects in test_data:
        one_hot = torch.zeros(num_classes)
        for obj in objects:
            if obj in object_to_idx:
                one_hot[object_to_idx[obj]] = 1.0
        conditions.append(one_hot)

    return torch.stack(conditions)
