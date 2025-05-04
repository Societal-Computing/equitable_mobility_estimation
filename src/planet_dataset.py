import os

import pandas as pd
import numpy as np
import rasterio
from torch.utils.data import Dataset

from utils import *
from logger import logger


class PlanetPatchDataset(Dataset):
    def __init__(self, im_data, transform=None, preprocess_type='none'):
        if isinstance(im_data, str):
            image_dataframe = pd.read_csv(im_data)
        elif isinstance(im_data, pd.DataFrame):
            image_dataframe = im_data.reset_index(drop=True)

        
        if 'label' not in image_dataframe.columns:
            image_dataframe['label'] = image_dataframe['day'].apply(lambda x: 0 if x.lower() == 'sunday' else 1)
        else:
            logger.info("Label column already present")

        print(image_dataframe.label.value_counts())
        self.image_dataframe = image_dataframe
        self.transform = transform
        self.preprocess_type = preprocess_type
        self.image_cache = {}

    def __len__(self):
        return len(self.image_dataframe)

    def stacking_rgb_images(self, image):
        if image.shape[0] == 8:
            red = image[5, :, :]  # Red band (e.g., Band 5 in some datasets)
            green = image[3, :, :]  # Green band
            blue = image[1, :, :]  # Blue band
        elif image.shape[0] == 4:
            red = image[2, :, :]  # Red band
            green = image[1, :, :]  # Green band
            blue = image[0, :, :]  # Blue band

        return np.dstack((red, green, blue)).astype('float32')
    def __getitem__(self, index):
        row = self.image_dataframe.iloc[index]
        image_path = row['image_path']

        with rasterio.open(image_path, 'r') as src:
            img_data = src.read()
            
        #img_data = self.image_cache[index]

        label = row['label']

        label = torch.tensor(label).long()

        if self.preprocess_type == 'derivatives':
            im = convert_to_image_derivatives(img_data)
        elif self.preprocess_type == 'rgb':
            im = convert_to_rgb(img_data)
            #im = self.stacking_rgb_images(img_data) #testing this feature against convert_to_rgb, doesn't work input supports unit8 only
        else:
            raise ValueError(
                "Invalid preprocess type, choose from ['derivatives', 'rgb']")

        if isinstance(im, torch.Tensor) and im.shape[0] > 3:
            im = im.numpy().astype(np.uint8)
            im_tensor = torch.stack([self.transform(im[i, :, :]) for i in range(im.shape[0])])
            im_tensor = im_tensor.squeeze(1)
        elif self.transform:
            im_tensor = self.transform(im)
        else:
            im_tensor = torch.as_tensor(im)

        return im_tensor, label
