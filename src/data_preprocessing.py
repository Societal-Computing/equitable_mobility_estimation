"""
This fle is to perform the data preprocessing steps for the parkinglot dataset.
The data preprocessing steps include the following for each parking lot:
  1. Removal of images with no clear pixels using UDM clear mask
  2. Removal of images by comparing them to mean histogram.

  UDM image has 8 channals;
    1. 0 - Clear
    2. 1 - Snow
    3. 2 - Cloud Shadow
    4. 3 - Light Haze
    5. 4 - Heavy Haze
    6. 5 - Cloud
    7. 6 - Confidence
    8. 7 - Unusable pixels


How to run:
    python src/data_preprocessing.py --geojson_base_dir dataset/parking_lots_to_query_new --masked_images_root dataset/Masked_Cluster_Images --save_path dataset
"""

import argparse
import re
from typing import Dict

import cv2
import geopandas as gpd
import pandas as pd
import random
import numpy as np
import rasterio
from rasterio.mask import mask
from skimage.transform import resize
from glob import glob

random.seed(42)
np.random.seed(42)

from logger import logger
from utils import *
import os


class DataPreProcessing:
    # use *_AnalyticMS_SR_8b_clip*.tif for filtering 8B images and *_AnalyticMS_SR_clip* for 4B images
    ANALYTIC_IMAGES_REGEX = '*_AnalyticMS_SR_*.tif'  # *_AnalyticMS_SR_8b_clip*.tif

    def __init__(self, all_shapefiles_root, masked_images_root, save_path, split_data) -> None:
        '''
        Parameters:
        all_shapefiles_root: str , this is the root directory where all the shapefiles are stored
        masked_images_root: str, this is the root directory where all the masked cluster images are stored
        save_path: str, this is the path where the filtered images are recoreded in a csv file
        '''
        self.all_shapefiles_root = all_shapefiles_root
        self.masked_images_root = masked_images_root
        self.save_path = save_path
        self.threshold = 0.2
        self.split_data = split_data

    @classmethod
    def remove_img_with_no_clear_udm(cls, udm_img, masked_area):
        '''
        checks if the image has no clear pixels in it then returns True else False
        True, means te image should be removed else it should be kept by returning False
        '''
        unique_values, value_counts = np.unique(udm_img[0], return_counts=True)
        clear_channel_value_counts_dict = dict(zip(unique_values, value_counts))

        masked_unique_values, masked_value_counts = np.unique(masked_area, return_counts=True)
        masked_value_counts_dict = dict(zip(masked_unique_values, masked_value_counts))

        if 1 in clear_channel_value_counts_dict.keys() and 1 in masked_value_counts_dict.keys():
            return False
        else:
            return True

    @classmethod
    def calculate_histogram(cls, image, bins=256):
        '''
        Calculate the histogram of the image
        '''
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    @classmethod
    def compare_histograms(cls, hist1: np.ndarray, hist2: np.ndarray):
        '''
        Compare the histograms of two images
        '''
        return np.sum(np.abs(hist1 - hist2))

    @classmethod
    def get_lotname_clustername(cls, shapefile_path: str): #TODO : changed this only for kyiv
        '''
        returns the parking lots name and cluster name from the shapefile path
        '''
        parking_lot_name = shapefile_path.stem
        cluster_name = shapefile_path.stem
        return parking_lot_name, cluster_name

    def cleaning_by_udm_single_shapefile(self, shapefile_path: str, masked_images_root: str):
        parking_lot_gdf = gpd.read_file(shapefile_path)

        parking_lot_name, cluster_name = self.get_lotname_clustername(shapefile_path)
        if 'parking' in parking_lot_name:
            masked_images_parent_dir = Path(masked_images_root).joinpath(cluster_name)#.joinpath(parking_lot_name) #change for nomral use
            logger.info(f"Processing for {cluster_name} - {parking_lot_name}")
            analytic_images_base_path = Path(masked_images_root).joinpath(cluster_name)#.joinpath(parking_lot_name)
        else:
            parking_lot_name = shapefile_path.stem
            masked_images_parent_dir = Path(masked_images_root).joinpath(f"{parking_lot_name}")
            cluster_name = parking_lot_name
            analytic_images_base_path = Path(masked_images_root).joinpath(cluster_name)

        all_analytic_images = list(analytic_images_base_path.glob(f"**\{self.ANALYTIC_IMAGES_REGEX}")) # Added ** to search in subdirectories
        logger.info(f"Total images for {cluster_name} - {parking_lot_name} are {len(all_analytic_images)}")
        filtered_data_ls = []
        for analytic_path in all_analytic_images:
            analytic_path = analytic_path.as_posix()
            udm_path = re.sub(r'AnalyticMS_SR_clip|AnalyticMS_SR_8b_clip', 'udm2_clip', analytic_path)
            if not Path(udm_path).exists():
                logger.info(f"UDM not found for {analytic_path}")
                continue

            with rasterio.open(udm_path) as udm_src:
                udm_img = udm_src.read()
                parking_lot_gdf = parking_lot_gdf.to_crs(udm_src.crs)
                masked_data, masked_transform = mask(udm_src, parking_lot_gdf.geometry, crop=True)

            is_remove_img = self.remove_img_with_no_clear_udm(udm_img, masked_data)

            if is_remove_img == False:
                filtered_data_ls.append({'cluster': cluster_name, 'image_path': analytic_path})

        filtered_data_df = pd.DataFrame(filtered_data_ls)
        logger.info(
            f"Total images after cleaning for {cluster_name} - {parking_lot_name} are {filtered_data_df.shape[0]}")
        return filtered_data_df

    @classmethod
    def get_reference_image_single_parkinglot(cls, images_path: str, shapefile_path: str):
        '''
        This function calculates the reference image for a single parking lot by taking the mean of all the gray images.
        '''
        all_images = []
        parking_lot_gdf = gpd.read_file(shapefile_path)
        for image_path in images_path:
            with rasterio.open(image_path) as src:
                image_8b = src.read()
                parking_lot_gdf = parking_lot_gdf.to_crs(src.crs)
                masked_data, _ = mask(src, parking_lot_gdf.geometry, crop=True)

            rgb_image = convert_to_rgb(image_8b)
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            masked_data = masked_data[0]
            masked_data = np.where(masked_data == 0, 0, 1)
            gray_image = gray_image * masked_data

            gray_image_resized = resize(gray_image, (32, 32))
            all_images.append(gray_image_resized)

        all_images_np = np.array(all_images)
        reference_image = all_images_np.mean(axis=0)
        return reference_image

    def cleaning_by_histogram_single_parkinglot(self, filtered_df_by_udm: Dict, shapefile_path: str):
        threshold = self.threshold
        filter_images_by_size = clean_data_by_size(filtered_df_by_udm['image_path'].tolist())
        reference_image = self.get_reference_image_single_parkinglot(filter_images_by_size, shapefile_path)
        images_path = filter_images_by_size
        parking_lot_name, cluster_name = self.get_lotname_clustername(shapefile_path)
        if reference_image.ndim == 2:
            reference_image = reference_image[np.newaxis, ...]

        reference_image = reference_image.astype(np.uint8)
        ref_hist = self.calculate_histogram(reference_image)
        filtered_images = []
        parking_lot_gdf = gpd.read_file(shapefile_path)

        for image_path in images_path:
            with rasterio.open(image_path) as src:
                image_8b = src.read()
                parking_lot_gdf = parking_lot_gdf.to_crs(src.crs)
                masked_data, _ = mask(src, parking_lot_gdf.geometry, crop=True)

            image = convert_to_rgb(image_8b)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            masked_data = masked_data[0]
            masked_data = np.where(masked_data == 0, 0, 1)
            image = image * masked_data

            image = image.astype(np.uint8)
            matched_image = image[np.newaxis, ...]
            hist = self.calculate_histogram(matched_image)
            similarity = self.compare_histograms(ref_hist, hist)

            if similarity < threshold:
                filtered_images.append(
                    {'cluster': cluster_name, 'parking_lot_name': parking_lot_name, 'image_path': image_path})

        filtered_images_df = pd.DataFrame(filtered_images)
        return filtered_images_df

    @staticmethod
    def split_data_train_test_val(all_filtered_images_df, save_path):
        unique_parkinglots = all_filtered_images_df['parking_lot_name'].unique()
        train_ratio = 0.85
        val_ratio = 0.1
        test_ratio = 0.05

        random.shuffle(unique_parkinglots)

        train_size = int(train_ratio * len(unique_parkinglots))
        val_size = int(val_ratio * len(unique_parkinglots))
        test_size = int(test_ratio * len(unique_parkinglots))

        train_parking_lot = unique_parkinglots[:train_size]
        val_parking_lot = unique_parkinglots[train_size: train_size + val_size]
        test_parking_lot = unique_parkinglots[train_size + val_size:]

        logger.info(f"Unique parking lots: {len(unique_parkinglots)}, Train parking lots: {len(train_parking_lot)}, Validation parking lots: {len(val_parking_lot)}, Test parking lots: {len(test_parking_lot)}")
        
        train_data = all_filtered_images_df[all_filtered_images_df['parking_lot_name'].isin(train_parking_lot)]
        val_data = all_filtered_images_df[all_filtered_images_df['parking_lot_name'].isin(val_parking_lot)]
        test_data = all_filtered_images_df[all_filtered_images_df['parking_lot_name'].isin(test_parking_lot)]

        train_data.to_csv(Path(save_path).joinpath('train_filtered.csv'), index=False)
        val_data.to_csv(Path(save_path).joinpath('val_filtered.csv'), index=False)
        test_data.to_csv(Path(save_path).joinpath('test_filtered.csv'), index=False)
        logger.info(f"Train, Validation and Test data saved at {save_path}")
    def main(self):
        all_shapefiles_root = self.all_shapefiles_root
        masked_images_root = self.masked_images_root
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)

        all_shapefiles = list(Path(all_shapefiles_root).glob('*.geojson'))
        
        all_filtered_images = []
        for shapefile_path in all_shapefiles:
            parking_lot_name, cluster_name = self.get_lotname_clustername(shapefile_path)
            if 'parking' in parking_lot_name:
                masked_images_parent_dir = Path(masked_images_root).joinpath(cluster_name)#.joinpath(parking_lot_name)
                logger.info(f"Processing for {cluster_name} - {parking_lot_name}")
            else:
                parking_lot_name = shapefile_path.stem
                masked_images_parent_dir = Path(masked_images_root).joinpath(f"{parking_lot_name}")
                cluster_name = parking_lot_name
            
            if not masked_images_parent_dir.exists() or len(list(masked_images_parent_dir.glob(f"**\{self.ANALYTIC_IMAGES_REGEX}"))) == 0:
                logger.info(f"Skipping {cluster_name} - {parking_lot_name} as no images found")
                continue
            logger.info(f"Cleaning data by UDM for {cluster_name} - {parking_lot_name}")
            filtered_df_by_udm = self.cleaning_by_udm_single_shapefile(shapefile_path, masked_images_root)
            if filtered_df_by_udm.shape[0] == 0:
                logger.info(f"No images found after UDM cleaning for {cluster_name} - {parking_lot_name}")
                continue
            logger.info(f"Cleaning data by histogram matching for {cluster_name} - {parking_lot_name}")
            filtered_images_df = self.cleaning_by_histogram_single_parkinglot(filtered_df_by_udm, shapefile_path)
            all_filtered_images.append(filtered_images_df)

        all_filtered_images_df = pd.concat(all_filtered_images)
        all_filtered_images_df['day'] = all_filtered_images_df['image_path'].apply(lambda x: find_day(Path(x).stem[:8]))
        all_filtered_images_df.to_csv(Path(save_path).joinpath('filtered_images_with_hist_matching.csv'), index=False)
        logger.info(f"Total filtered images are {all_filtered_images_df.shape[0]}")
        if self.split_data:
            logger.info("Splitting data into train, test and validation")
            self.split_data_train_test_val(all_filtered_images_df, save_path)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessing for Parking Lot Dataset', usage='python data_preprocessing.py --geojson_base_dir dataset/parking_lots_to_query_new --masked_images_root dataset/Masked_Cluster_Images --save_path dataset --split_data True')
    parser.add_argument('--geojson_base_dir', type=str, help='Root directory where all the shapefiles are stored')
    parser.add_argument('--masked_images_root', type=str,
                        help='Root directory where all the masked cluster images are stored')
    parser.add_argument('--save_path', type=str, help='Path where the filtered images are recorded in a csv file')
    parser.add_argument('--split_data', type=str, default=False, help='Split the data into train, test and validation')
    args = parser.parse_args()
    geojson_base_dir = args.geojson_base_dir
    masked_images_root = args.masked_images_root
    save_path = args.save_path
    split_data = args.split_data

    split_data = True if split_data.lower() == 'true' else False

    data_preprocessing = DataPreProcessing(geojson_base_dir, masked_images_root, save_path, split_data)
    data_preprocessing.main()
