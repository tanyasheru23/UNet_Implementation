import os
from PIL import Image
import pathlib

from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, key: int):
        image_path = os.path.join(self.image_dir, self.images[key])
        # mask_path = os.path.join(self.mask_dir, self.masks[key])
        mask_path = os.path.join(self.mask_dir, self.images[key].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(image_path).convert("RGB")) # we are using np array since we will be using Albumentations library which req np array
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        
        if self.transform:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask