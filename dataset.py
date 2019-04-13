import os
import glob
import torch
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

def rescale(x):
    return (x * 0.5) + 0.5

class CityScapesDataset(Dataset):
    def __init__(self, labels, scale, phase, maximum=None):
        super().__init__()
        self.scale = scale
        self.labels = labels
        self.rgb_paths = sorted(glob.glob(os.path.join('/scratch/gobi1/chuhang/CityScapes/leftImg8bit_trainvaltest', 'leftImg8bit', phase, '*/*.png')))
        self.label_paths = sorted(glob.glob(os.path.join('/scratch/gobi1/chuhang/CityScapes/gtFine_trainvaltest', 'gtFine', phase, '*/*labelIds.png')))
        self.img_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

        if not maximum is None and maximum < len(self.rgb_paths):
            self.rgb_paths = self.rgb_paths[:maximum]
            self.label_paths = self.label_paths[:maximum]

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        gt = Image.open(self.rgb_paths[index])
        w, h = gt.size

        # Crop the ground truth if scale index results in non-integer
        if w % self.scale != 0 or h % self.scale != 0:
            w, h = (int((w // self.scale) * self.scale), int((h // self.scale) * self.scale))
            gt = gt.crop((0, 0, w, h))

        # Produce the LR image with bicubic resampling
        lr = gt.resize((w // self.scale, h // self.scale), resample=Image.BICUBIC)
        gt = self.img_tfm(gt)
        lr = self.img_tfm(lr)

        # Mask is not needed if no labels are specified
        if self.labels is None:
            return lr, gt

        else:
            labeled_img = Image.open(self.label_paths[index]).crop((0, 0, w, h))
            labeled_img = np.array(labeled_img)
            mask = np.zeros((h, w)).astype(np.bool)
            # Add each specified label to the mask
            for label in self.labels:
                mask = np.logical_or(mask, labeled_img == int(label))

            mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze_(0)
            return lr, gt, mask
