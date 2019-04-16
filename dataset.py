import os
import glob
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import math
import skimage

from torch.utils.data import Dataset
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import measurements

def rescale(x):
    return (x * 0.5) + 0.5

class CityScapesDataset(Dataset):
    def __init__(self, labels, scale, phase, crop_size=None, maximum=None):
        super().__init__()
        self.scale = scale
        self.labels = labels
        self.crop_size = crop_size
        self.rgb_paths = sorted(glob.glob(os.path.join('/scratch/gobi1/chuhang/CityScapes/leftImg8bit_trainvaltest', 'leftImg8bit', phase, '*/*.png')))
        self.label_paths = sorted(glob.glob(os.path.join('/scratch/gobi1/chuhang/CityScapes/gtFine_trainvaltest', 'gtFine', phase, '*/*labelIds.png')))
        self.img_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

        if not maximum is None and maximum < len(self.rgb_paths):
            self.rgb_paths = self.rgb_paths[:maximum]
            self.label_paths = self.label_paths[:maximum]

        # Get valid crop locations for all images, if crop size and labels are needed
        if not self.labels is None and not self.crop_size is None:
            self.create_valid_images()

    def __len__(self):
        return len(self.rgb_paths)

    def get_LR_and_GT(self, gt):
        # Create LR
        lr_w, lr_h = gt.size[0] // self.scale, gt.size[1] // self.scale
        lr = gt.resize((lr_w, lr_h), resample=Image.BICUBIC)
        # Adjust GT if initial size incompatible with scale
        new_w, new_h = lr_w * self.scale, lr_h * self.scale
        if new_w != gt.size[0] or new_h != gt.size[0]:
            gt = gt.crop((0, 0, new_w, new_h))
        return lr, gt

    def __getitem__(self, index):
        gt = Image.open(self.rgb_paths[index])
        # Mask is not needed if no labels are specified
        if self.labels is None:
            # Randomly crop image to desired size
            if not self.crop_size is None:
                new_width, new_height = gt.size[0]-self.crop_size, gt.size[1]-self.crop_size
                rand_index = random.randint(0, new_height * new_width)
                top_left_y, top_left_x = rand_index % new_height, rand_index // new_height
                gt = gt.crop((top_left_x, top_left_y, top_left_x + self.crop_size, top_left_y + self.crop_size))
            # Return LR and GT images only
            lr, gt = self.get_LR_and_GT(gt)
            return self.img_tfm(lr), self.img_tfm(gt)

        # Fetch semantic labels
        label_img = Image.open(self.label_paths[index])

        # Self.valid_masks is None when no pixels correspond to semantic labels
        # This case is handled in the main training loop
        if not self.crop_size is None and not self.valid_masks[index] is None:
            # Get set of all valid crop locations
            valid_indices = self.valid_masks[index].nonzero()
            offset = self.offsets[index]
            # Randomly select valid crop location
            rand_index = random.randint(0, valid_indices[0].shape[0])
            top_left = offset[0] + valid_indices[0][rand_index], offset[1] + valid_indices[1][rand_index]
            # Crop images at chosen location
            gt = gt.crop((top_left[1], top_left[0], top_left[1] + self.crop_size, top_left[0] + self.crop_size))
            label_img = label_img.crop((top_left[1], top_left[0], top_left[1] + self.crop_size, top_left[0] + self.crop_size))

        # Obtain LR image and adjust sizes if necessary
        lr, gt = self.get_LR_and_GT(gt)
        if gt.size[0] != label_img.size[0] or gt.size[1] != label_img.size[1]:
            label_img = label_img.crop((0, 0, gt.size[0], gt.size[1]))

        # Add each specified semantic label to the mask as ones
        label_img = np.array(label_img)
        mask = np.zeros_like(label_img).astype(np.bool)
        for label in self.labels:
            mask = np.logical_or(mask, label_img == int(label))

        return self.img_tfm(lr), self.img_tfm(gt), torch.from_numpy(mask.astype(np.float32)).unsqueeze_(0)

    def create_valid_images(self):
        self.valid_masks = [None] * len(self.label_paths)
        self.offsets = [None] * len(self.label_paths)
        # Threshold value
        k = (self.crop_size * self.crop_size) * 0.001

        for i in tqdm(range(len(self.label_paths))):
            img = np.array(Image.open(self.label_paths[i]))
            # Create semantic mask
            mask = np.zeros_like(img)
            for label in self.labels:
                mask = np.logical_or(mask, img == label)

            # Skip if mask has no valid pixels
            num_pixels = np.sum(mask)
            if num_pixels == 0:
                continue

            # Find tightest possible bounding box for all mask pixels
            result = measurements.find_objects(mask)[0]
            start_y, stop_y = result[0].start, result[0].stop
            start_x, stop_x = result[1].start, result[1].stop

            # Only check values in or immediately surrounding bounding box
            # expand the bounding box depending on desired crop size
            half_crop = math.ceil(self.crop_size / 2)
            if stop_y + half_crop > mask.shape[0]:
                start_y = max(start_y - self.crop_size, 0)
            elif start_y - half_crop < 0:
                stop_y = min(stop_y + self.crop_size, mask.shape[0])
            else:
                start_y -= half_crop
                stop_y += half_crop

            if stop_x + half_crop > mask.shape[1]:
                start_x = max(start_x - self.crop_size, 0)
            elif start_x - half_crop < 0:
                stop_x = min(stop_x + self.crop_size, mask.shape[1])
            else:
                start_x -= half_crop
                stop_x += half_crop

            # Reduce semantic mask to approximate bounding box
            mask = mask[start_y : stop_y, start_x : stop_x]
            # Compute integral image for quick access to pixel coverage info
            integral_img = skimage.transform.integral.integral_image(mask)
            # Get top left indices of potential crop locations
            indices = np.indices((integral_img.shape[0]-self.crop_size, integral_img.shape[1]-self.crop_size))
            # Only accept indices where mask pixels sum to more than k
            valid_mask = integral_img[indices[0] + self.crop_size, indices[1] + self.crop_size] - integral_img[indices[0] + self.crop_size, indices[1]] - integral_img[indices[0], indices[1] + self.crop_size] + integral_img[indices[0], indices[1]] > k

            assert np.sum(valid_mask) > 0, "No possible crop satisfying threshold in {}".format(self.label_paths[i])
            # Cache mask and offset coords for valid crop locations
            self.valid_masks[i] = valid_mask
            self.offsets[i] = (start_y, start_x)


if __name__ == "__main__":
    d = CityScapesDataset([26], 3, 'train', crop_size=512, maximum=10)
    # for i in tqdm(range(len(d))):
    out = d.__getitem__(5)
    save_image(rescale(out[0]), 'lr.png')
    save_image(rescale(out[1]), 'temp.png')
    save_image(rescale(out[2]), 'mask.png')
