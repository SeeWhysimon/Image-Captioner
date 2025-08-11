import os
import json
from PIL import Image
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(self, image_dir, ann_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(ann_path, 'r') as f:
            data = json.load(f)
        self.images = data['images']
        self.annotations = data['annotations']

        self.id2captions = {}
        for ann in self.annotations:
            self.id2captions.setdefault(ann['image_id'], []).append(ann['caption'])

        self.samples = []
        for img in self.images:
            img_path = os.path.join(image_dir, img['file_name'])
            captions = self.id2captions.get(img['id'], [])
            for caption in captions:
                self.samples.append((img_path, caption))

    
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption


class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_path