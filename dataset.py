import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json

TRAINING_DATA_DIR = 'dataset/train'
TESTING_DATA_DIR = "dataset/test"
VALIDATION_DATA_DIR = "dataset/valid"

# Load the JSON file containing data information
def load(DATA_DIR):
    with open(os.path.join(DATA_DIR, '_annotations.coco.json'), 'r') as f:
        data_info = json.load(f)
    return data_info

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.data_info = load(data_dir)
        self.transform = transform
        self.image_ids = list(self.data_info['images'])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]['id']
        img_path = os.path.join(self.data_dir, self.data_info['images'][idx]['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for the current image
        annotations = [ann for ann in self.data_info['annotations'] if ann['image_id'] == img_id]
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
        
        target = {"boxes": boxes, "labels": labels}

        return image, target, img_path