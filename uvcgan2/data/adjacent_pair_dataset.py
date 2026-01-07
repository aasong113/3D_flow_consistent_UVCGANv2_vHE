# uvcgan2/data/adjacent_pair_dataset.py

import os
import glob
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AdjacentZPairDataset(Dataset):
    def __init__(self, root_dir, z_spacing=1, transform=None):
        self.root_dir = root_dir
        self.z_spacing = z_spacing
        self.transform = transform or transforms.ToTensor()
        self.img_dict = self.build_img_dict()

        # Build valid (Z, P) pairs where a match exists
        self.pairs = self.build_pairs()

    def build_img_dict(self):
        files = glob.glob(os.path.join(self.root_dir, '*.tif'))
        pattern = re.compile(r'img=(\d+)_P=(\d+)\.tif')
        img_dict = {}

        for f in files:
            match = pattern.search(os.path.basename(f))
            if match:
                z, p = int(match.group(1)), int(match.group(2))
                img_dict[(z, p)] = f
        return img_dict

    def build_pairs(self):
        pairs = []
        for (z, p) in self.img_dict:
            forward = (z + self.z_spacing, p)
            backward = (z - self.z_spacing, p)

            if forward in self.img_dict:
                pairs.append(((z, p), forward))
            elif backward in self.img_dict:
                pairs.append(((z, p), backward))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (z1, p), (z2, _) = self.pairs[idx]
        
        path1 = self.img_dict[(z1, p)]
        path2 = self.img_dict[(z2, p)]

        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        return {
            'A': self.transform(img1),
            'B': self.transform(img2),
            'A_name': os.path.basename(path1),
            'B_name': os.path.basename(path2),
            'meta': {'img1': (z1, p), 'img2': (z2, p)}
        }

