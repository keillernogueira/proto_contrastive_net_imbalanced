import os
import numpy as np

from PIL import Image

import imageio
from skimage import img_as_float
from skimage import transform

import torch
from torch.utils import data

from dataloaders.data_utils import create_distrib, create_or_load_statistics, \
    normalize_images, create_distrib_knn, data_augmentation

Image.MAX_IMAGE_PIXELS = None


class ISPRSDataLoader(data.Dataset):
    def __init__(self, mode, dataset, dataset_input_path, images, crop_size, stride_size,
                 mean=None, std=None, output_path=None):
        super().__init__()
        assert mode in ['Train', 'Validation']

        self.mode = mode
        self.dataset = dataset
        self.dataset_input_path = dataset_input_path
        self.images = images
        self.crop_size = crop_size
        self.stride_size = stride_size
        self.num_classes = 2
        self.output_path = output_path

        self.data, self.labels = self.load_images()

        self.distrib, self.gen_classes = self.make_dataset()

        if mean is None and std is None:
            self.mean, self.std = create_or_load_statistics(self.data, self.distrib, self.crop_size,
                                                            self.stride_size, output_path)
        elif mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:  # MS coco mean and std
            self.mean = np.asarray([0.485, 0.456, 0.406])
            self.std = np.asarray([0.229, 0.224, 0.225])

        if len(self.distrib) == 0:
            raise RuntimeError('Found 0 samples, please check the data set path')

    def encode_mask(self, msk):
        msk = msk.astype(np.int64)
        new = np.zeros((msk.shape[0], msk.shape[1]), dtype=int)

        msk = msk // 255
        msk = msk * (1, 7, 49)
        msk = msk.sum(axis=2)

        new[msk == 1 + 7 + 49] = 0  # Street.
        new[msk ==         49] = 0  # Building.
        new[msk ==     7 + 49] = 0  # Grass.
        new[msk ==     7     ] = 0  # Tree.
        new[msk == 1 + 7     ] = 1  # Car.
        new[msk == 1         ] = 0  # Surfaces.

        return new

    def load_images(self):
        images = []
        masks = []
        for img in self.images:
            print('--------------' + str(img) + '------------------')
            t_image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, 'images',
                                                               'top_mosaic_09cm_area' + str(img) + '.tif')))

            t_mask = imageio.imread(os.path.join(self.dataset_input_path, 'masks',
                                                 'top_mosaic_09cm_area' + str(img) + '.tif'))
            print(t_image.shape, t_mask.shape)

            encoded_mask = self.encode_mask(t_mask)
            print(encoded_mask.shape, np.unique(encoded_mask), np.bincount(encoded_mask.flatten()))

            images.append(t_image)
            masks.append(encoded_mask)

        return images, masks

    def make_dataset(self):
        distrib, gen_classes = create_distrib(self.labels, self.crop_size, self.stride_size,
                                              self.num_classes, self.dataset, return_all=True)

        return distrib, gen_classes

    def __getitem__(self, index):
        cur_map, cur_x, cur_y = self.distrib[index][0], self.distrib[index][1], self.distrib[index][2]

        img = np.copy(self.data[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size, :])
        label = np.copy(self.labels[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size])

        # Normalization.
        normalize_images(img, self.mean, self.std)

        if 'Train' in self.mode or 'train' in self.mode:
            img, label = data_augmentation(img, label)

        img = np.transpose(img, (2, 0, 1))

        # Turning to tensors.
        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(label.copy())

        # Returning to iterator.
        return img.float(), label, cur_map, cur_x, cur_y

    def __len__(self):
        return len(self.distrib)
