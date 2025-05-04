from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import rioxarray
import seaborn as sns
import torch
import yaml
import json
import geopandas as gpd
from PIL import Image
from matplotlib import pyplot as plt
from skimage.util import view_as_blocks
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torchvision import transforms

from logger import logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



CRS = "EPSG:4326"

def load_config(config_path):
    '''Reads the config file and returns the dictionary'''
    if isinstance(config_path, str):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    elif isinstance(config_path, dict):
        config = config_path
    else:
        raise ValueError("Invalid config path or config dictionary")
    return config

def file_reader(input_file_path):
    try:
        input_file = None
        if Path(input_file_path).suffix in [".csv", ".xlsx"]:
            input_file = pd.read_csv(input_file_path)
        if Path(input_file_path).suffix in [".gpkg", ".shp"]:
            input_file = gpd.read_file(input_file_path)
        return input_file
    except Exception as e:
        raise f"{Path(input_file_path).suffix} not a support file format"

def read_geom(file_path):
    with open(file_path) as f:
        bbox = json.load(f)['features'][0]['geometry']
        return bbox

def generate_experiment_name(model_name):
    ''' generate experiment name '''
    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = Path("experiments").joinpath(str(exp_name) + "_" + model_name)
    experiment_name.mkdir(parents=True, exist_ok=True)
    return experiment_name


def get_transforms(augmentation):
    ''' load transformations from config file'''
    x = [getattr(transforms, k)(**v) for k, v in augmentation.items()]
    return transforms.Compose(x)


def save_inference_config(config, save_path, train_type):
    ''' save inference config '''
    if train_type == 'single':
        model_name = config['model_config']['model_name']
        weight_path = save_path.joinpath(f'{model_name}.pt')
        config['model_config']['model_weight'] = weight_path.as_posix()
        config['train_config']['train_type'] = train_type

    elif train_type == 'end2end':
        reg_model_name = config['model_config']['regression']['model_name']
        reg_weight_path = save_path.joinpath(f'{reg_model_name}.pt')
        config['model_config']['regression']['model_weight'] = reg_weight_path.as_posix()

        cl_model_name = config['model_config']['classification']['model_name']
        cl_weight_path = save_path.joinpath(f'{cl_model_name}.pt')
        config['model_config']['classification']['model_weight'] = cl_weight_path.as_posix()
        config['train_config']['train_type'] = train_type

    inference_config = save_path.joinpath('inference_config.yaml')
    if isinstance(config['data_config']['train_data']['data_path'], pd.DataFrame):
        config['data_config']['train_data']['data_path'] = 'filtered_with_min_lot_size'
    if isinstance(config['data_config']['test_data']['data_path'], pd.DataFrame):
        config['data_config']['test_data']['data_path'] = 'filtered_with_min_lot_size'

    with open(inference_config.as_posix(), 'w') as f:
        yaml.dump(config, f, indent=4, sort_keys=False)

    logger.info(f"model info saved to {inference_config.as_posix()}")


def check_size_most(img_path):
    ''' Finding the most occuring size of the image in the dataset'''
    img_size = []
    for i in img_path:
        img = rioxarray.open_rasterio(i)
        img_size.append(img.shape)
    return Counter(img_size).most_common(1).pop()[0]


def clean_data(image_list: List[str]):
    '''This function will clean the data by removing the images with less
    data pixels and non-data pixels'''
    logger.info(f"Number of Images before cleaning: {len(image_list)}")
    
    image_list = clean_data_by_size(image_list)
    logger.info(f"Number of Images after cleaning by size: {len(image_list)}")

    image_list = clean_data_by_black_pixels(image_list)
    logger.info(f"Number of Images after removal of black pixels: {len(image_list)}")

    image_list = clean_data_by_white_pixels(image_list)
    logger.info(f"Number of Images after removal of white pixels: {len(image_list)}")

    return image_list


def clean_data_by_size(img_path: List[str]):
    '''Cleaning the data by removing the images which are not of the most common size'''
    logger.info(f"Initial size of dataset: {len(img_path)}")
    size = check_size_most(img_path)
    logger.info(f"Most common size of the dataset: {size}")
    for i in img_path:
        img = rioxarray.open_rasterio(i)
        if img.shape != size:
            img_path.remove(i)
    logger.info(f"Final size of dataset(after removing faulty size images): {len(img_path)}")
    return img_path


def clean_data_by_black_pixels(img_path_list):
    black_pixels = []
    for i in img_path_list:
        img = rioxarray.open_rasterio(i)
        black_pixels.append(np.count_nonzero(img.data[0:3].transpose(1, 2, 0) == 0))
    img_path_median = []
    for i in range(len(img_path_list)):
        if black_pixels[i] <= np.median(black_pixels):
            img_path_median.append(img_path_list[i])
    logger.info(f"Final size of dataset(after removing faulty black pixels in images): {len(img_path_median)}")
    return img_path_median


def clean_data_by_white_pixels(img_path_list, threshold=200):
    white_pixels = []
    for i in img_path_list:
        img = rioxarray.open_rasterio(i)
        white_pixels.append(np.count_nonzero(img.data[0:3].transpose(1, 2, 0) >= threshold))

    img_path_mean_white = []
    for i in range(len(img_path_list)):
        if white_pixels[i] <= np.mean(white_pixels):
            img_path_mean_white.append(img_path_list[i])
    logger.info(f"Final size of dataset(after removing faulty white pixels in images) {len(img_path_mean_white)}")
    return img_path_mean_white


def find_day(date_string):
    date_string = f"{date_string[:4]}-{date_string[4:6]}-{date_string[6:8]}"
    try:
        date = datetime.strptime(date_string, "%Y-%m-%d")
        day = date.strftime("%A")
        return day
    except ValueError:
        return "Invalid date format. Please provide the date in YYYY-MM-DD format."


def patchify(image_array, patch_size):
    num_patches_y = int(np.ceil(image_array.shape[0] / patch_size))
    num_patches_x = int(np.ceil(image_array.shape[1] / patch_size))

    pad_y = num_patches_y * patch_size - image_array.shape[0]
    pad_x = num_patches_x * patch_size - image_array.shape[1]

    padded_image = np.pad(image_array, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
    
    patches = view_as_blocks(padded_image, block_shape=(patch_size, patch_size, image_array.shape[2]))
    patches = patches.reshape(-1, patch_size, patch_size, image_array.shape[2])
    return patches

# 8B image processing functions
# def normalise_band(band):
#     band = (band - band.min()) / ((band.max() - band.min()) + 1e-8)
#     return band * 255


# def brighten(band):
#     alpha = 0.13
#     beta = 0
#     return np.clip(alpha * band + beta, 0, 255)


# def gammacorr(band):
#     gamma = 2
#     return np.power(band, 1 / gamma)


# def convert_to_rgb_mixin(image):
#     if image.shape[0] == 8:
#         red = image[5, :, :]
#         green = image[3, :, :]
#         blue = image[1, :, :]
#         nir = image[7, :, :]
#     elif image.shape[0] == 4:
#         red = image[2, :, :]
#         green = image[1, :, :]
#         blue = image[0, :, :]
#         nir = image[3, :, :]

#     red = brighten(red)
#     green = brighten(green)
#     blue = brighten(blue)
#     nir = brighten(nir)

#     red = normalise_band(red)
#     green = normalise_band(green)
#     blue = normalise_band(blue)
#     nir = normalise_band(nir)

#     return red, green, blue, nir


# def convert_to_rgb(image):
#     red, green, blue, _ = convert_to_rgb_mixin(image)
#     rgb = np.dstack((red, green, blue))
#     return rgb.astype(np.uint8)


def normalize_band(band, global_min, global_max):
    """Normalize a band using global min and max values."""
    #norm_value = (band - global_min) / (global_max - global_min + 1e-8)
    norm_value = band / (global_max + 1e-6)
    return norm_value

def apply_gamma_correction(band, gamma=2.2):
    """Apply gamma correction to a band."""
    return np.power(band, 1 / gamma)

def convert_to_rgb(image, dtype=np.uint8):
    """
    Convert 4-band or 8-band orthorectified image to RGB.
    Handles normalization, gamma correction, and scaling.
    """
    # Check if image has 4 or 8 bands
    if image.shape[0] == 8:
        red = image[5, :, :]  # Red band (e.g., Band 5 in some datasets)
        green = image[3, :, :]  # Green band
        blue = image[1, :, :]  # Blue band
    elif image.shape[0] == 4:
        red = image[2, :, :]  # Red band
        green = image[1, :, :]  # Green band
        blue = image[0, :, :]  # Blue band
    elif image.shape[0] == 3:
        red = image[0, :, :]
        green = image[1, :, :]
        blue = image[2, :, :]
    else:
        raise ValueError("Unsupported number of bands. Only 4-band or 8-band images are supported.")

    # Apply gamma correction
    # red = apply_gamma_correction(red)
    # green = apply_gamma_correction(green)
    # blue = apply_gamma_correction(blue)
    # Normalize each band using the global min and max
    global_min = np.min([red.min(), green.min(), blue.min()])
    global_max = np.max([red.max(), green.max(), blue.max()])
    
    red = normalize_band(red , global_min, global_max)
    green = normalize_band(green , global_min, global_max)
    blue = normalize_band(blue , global_min, global_max)

    #Scale to 8-bit range
    red = (red * 256).clip(0, 256).astype(dtype)
    green = (green * 256).clip(0, 256).astype(dtype)
    blue = (blue * 256).clip(0, 256).astype(dtype)
    # Combine into an RGB image
    rgb = np.dstack((red, green, blue))
    return rgb


def convert_to_rgbn(image):
    red, green, blue, nir = convert_to_rgb_mixin(image)
    rgbn = np.dstack((red, green, blue, nir))
    return rgbn.astype(np.uint8)


def convert_to_image_derivatives(image):
    for i in range(image.shape[0]):
        image[i, :, :] = get_gradient_magnitude(image[i, :, :])

    if image.shape[0] > 3:
        image = torch.Tensor(image)
    else:
        image = image.astype(np.uint8)

    return image


def pan_sharpen(rgb_image, gray_image):
    new_red = rgb_image[:, :, 0] / ((rgb_image[:, :, 0] + rgb_image[:, :, 1] + rgb_image[:, :, 2]) * gray_image)
    new_green = rgb_image[:, :, 1] / ((rgb_image[:, :, 0] + rgb_image[:, :, 1] + rgb_image[:, :, 2]) * gray_image)
    new_blue = rgb_image[:, :, 2] / ((rgb_image[:, :, 0] + rgb_image[:, :, 1] + rgb_image[:, :, 2]) * gray_image)
    new_rgb_image = np.stack([new_red, new_green, new_blue], axis=2)
    return new_rgb_image


def create_density_map_luminance(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    density_map = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return density_map


def compute_fourier_transform_cv2(channel):
    dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = np.log(magnitude_spectrum + 1)
    return magnitude_spectrum


def get_gradient_magnitude(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_x = np.abs(gradient_x)

    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_y = np.abs(gradient_y)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return gradient_magnitude_normalized


def extract_descriptors(batch_data):
    logger.info("Extracting SIFT descriptors...")
    output_descriptors = torch.zeros((batch_data.shape[0], 128), dtype=torch.float32)

    for i in range(batch_data.shape[0]):
        data = batch_data[i].cpu().numpy().astype(np.uint8)

        if data.shape[0] > 3:
            img = convert_to_rgb(data)
        elif data.shape[0] == 3:
            img = data.transpose(1, 2, 0)
        else:
            img = data.astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        #print(f"keypoints: {len(keypoints)}")

        if descriptors is not None:
            if descriptors.shape[0] > 128:
                descriptors = descriptors[:128]
            elif descriptors.shape[0] < 128:
                padding = np.zeros((128 - descriptors.shape[0], descriptors.shape[1]))
                descriptors = np.vstack((descriptors, padding))

            output_descriptors[i] = torch.from_numpy(descriptors).reshape(-1)[:128]

    logger.info(f"SIFT descriptors extracted with size {output_descriptors.shape}...")
    output_descriptors = output_descriptors.to(device)
    return output_descriptors


def plot_loss(num_epochs, epoch_train_loss, epoch_val_loss, experiment_dir):
    plt.plot(range(num_epochs), epoch_train_loss, label='train loss')
    plt.plot(range(num_epochs), epoch_val_loss, label='val loss')
    plt.legend()
    plt.savefig(experiment_dir.joinpath('loss.png'))


def plot_gt_pred(epoch_train_true, epoch_train_pred, epoch_val_true, epoch_val_pred, experiment_dir):
    plt.plot(epoch_train_true, label='train true')
    plt.plot(epoch_train_pred, label='train pred')
    plt.plot(epoch_val_true, label='val true')
    plt.plot(epoch_val_pred, label='val pred')
    plt.legend()
    plt.savefig(experiment_dir.joinpath('gt_pred_plot.png'))


def plot_confusion_matrix(y_true, y_pred, labels, experiment_dir, plot_name='confusion_matrix.png'):
    y_pred = np.argmax(y_pred, axis=-1)
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(experiment_dir.joinpath(plot_name))
    plt.close()


def plot_auc_roc_curve(y_true, y_pred, experiment_dir, plot_name='roc_auc_curve.png'):
    n_classes = len(np.unique(y_true))
    y_true = y_true.astype(int)
    # print(y_true, n_classes)
    if n_classes <= 2:
        try:
            y_pred = y_pred[:, 1]
        except Exception as e:
            y_pred = y_pred

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Calculate the AUC
        roc_auc = auc(fpr, tpr)

        # Plotting
        plt.figure()
        lw = 2  # Line width
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc="lower right", fontsize=14)
        plt.savefig(experiment_dir.joinpath(plot_name))
        return roc_auc

    elif n_classes > 2:
        n_classes = len(np.unique(y_true))
        # Binarize the true labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC AUC
        macro_roc_auc = roc_auc_score(y_true_bin, y_pred, average='macro')

        # Plot ROC curves
        plt.figure()
        colors = ['blue', 'red']  # Example colors for different classes
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(experiment_dir.joinpath(plot_name))
        plt.close()
        return macro_roc_auc


def plots(num_epochs, epoch_train_loss, epoch_test_loss, epoch_train_true, epoch_train_pred, epoch_test_true,
          epoch_test_pred, experiment_dir, task):
    epoch_train_true = np.array(epoch_train_true) # of shape epoch,length,label
    #print(f"epoch train true: {epoch_train_true.shape}")
    epoch_train_true = np.mean(epoch_train_true, axis=0) # should be changed to be loaded from the dataloader
    epoch_train_pred = np.array(epoch_train_pred)
    #print(f"epoch train pred: {epoch_train_pred.shape}")
    epoch_train_pred = np.mean(epoch_train_pred, axis=0)
    epoch_test_true = np.array(epoch_test_true)
    if epoch_test_true.ndim == 3:
        epoch_test_true = np.mean(epoch_test_true, axis=0)

    epoch_test_pred = np.array(epoch_test_pred)
    print(f"epoch test pred : {epoch_test_pred.shape}")
    epoch_test_pred = np.mean(epoch_test_pred, axis=0)

    if task == 'regression':
        plot_loss(num_epochs, epoch_train_loss, epoch_test_loss, experiment_dir)
        logger.info("Loss plot saved")

        plot_gt_pred(epoch_train_true, epoch_train_pred, epoch_test_true, epoch_test_pred, experiment_dir)
        logger.info("Loss plot saved")
    elif task == 'classification':
        labels = np.unique(epoch_train_true)
        plot_confusion_matrix(epoch_test_true, epoch_test_pred, labels, experiment_dir)
        roc_auc = plot_auc_roc_curve(epoch_test_true, epoch_test_pred, experiment_dir)
        logger.info(f"AUC: {roc_auc}")


def inference_plots(y_true, y_pred, experiment_dir, task, plot_name='image_level'):
    if y_true.ndim == 1:
        y_tue = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    if task == 'regression':
        plt.figure()
        plt.plot(y_true, label='true')
        plt.plot(y_pred, label='pred')
        plt.legend(fontsize=12)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.savefig(experiment_dir.joinpath(f'{plot_name}_inference_occupancy.png'))
        plt.close()
        return None
    elif task == 'classification':
        labels = np.unique(y_true)
        plot_confusion_matrix(y_true, y_pred, labels, experiment_dir, plot_name + '_inference_confusion_matrix.png')
        roc_auc = plot_auc_roc_curve(y_true, y_pred, experiment_dir, plot_name + '_inference_roc_auc_curve.png')
        logger.info(f"AUC: {roc_auc}")
        return roc_auc
