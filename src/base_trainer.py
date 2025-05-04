from typing import Dict
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import *
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

from planet_dataset import PlanetPatchDataset
from logger import logger


class EarlyStopping:
    def __init__(self, patience: int, save_path: str, model_name: str, delta: int = 0, verbose: bool =False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = Path(save_path).joinpath(model_name + '.pt')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(
                f'Validation loss has decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class BaseTrainer:
    def __init__(self, config_path: str, early_stopping=True) -> None:
        self.config = self.load_config(config_path)
        self.data_config = self.config['data_config']
        self.model_config = self.config['model_config']
        self.train_config = self.config['train_config']
        self.early_stopping = early_stopping

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load_config(config_path):
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        elif isinstance(config_path, dict):
            config = config_path
        else:
            raise ValueError("config_path should be either a string or a dictionary")
        return config


    def check_remove(row):
        '''
        We expect Sunday image to have value close to mean of reference image
        and Saturday image to have value greater than mean of reference image

        return:
        1 if image is to be removed else 0
        '''
        sun_threshold = 0.03
        sat_threshold = 0.09
        day = row['day']
        if day.lower() == 'sunday':
            return 1 if abs(row['image_seb_grad_mean'] - row['ref_image_seb_grad_mean']) > sun_threshold else 0
        elif day.lower() == 'saturday':
            return 1 if abs(row['image_seb_grad_mean'] - row['ref_image_seb_grad_mean']) < sat_threshold else 0
    

    def get_dataloader(self, train_data_path: str, val_data_path: str, train_augementations: Dict, val_augmentations: Dict, preprocess_type: str, batch_size: int, dimensionless: bool = False):
        train_transform = get_transforms(train_augementations)
        val_transform = get_transforms(val_augmentations)

        if isinstance(train_data_path, str):
            train_df = pd.read_csv(train_data_path)
        elif isinstance(train_data_path, pd.DataFrame):
            train_df = train_data_path.copy()
        else:
            raise ValueError("train_data_path should be either a string or a dataframe")

        if isinstance(val_data_path, str):
            val_df = pd.read_csv(val_data_path)
        elif isinstance(val_data_path, pd.DataFrame):
            val_df = val_data_path.copy()
        else:
            raise ValueError("val_data_path should be either a string or a dataframe")

        train_dataset = PlanetPatchDataset(train_df, train_transform, preprocess_type)
        test_dataset = PlanetPatchDataset(val_df, val_transform, preprocess_type)

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")

        if dimensionless:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self.pad_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self.pad_collate_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False, num_workers=0)

        return train_loader, test_loader

    def pad_collate_fn(self, batch):
        images, labels = zip(*batch)
        
        # Find maximum height and width in this batch
        max_height = max(img.shape[1] for img in images)
        max_width = max(img.shape[2] for img in images)
        logger.info(f"Max height: {max_height}, Max width: {max_width}")
        
        padded_images = []
        #anchor_pair_padded_images = []

        for img in images:
            pad_width = max_width - img.shape[2]
            pad_height = max_height - img.shape[1]
            padded_img = F.pad(img, (0, pad_width, 0, pad_height), mode='constant', value=0)
            padded_images.append(padded_img)
        
        padded_images_tensor = torch.stack(padded_images)
        labels = torch.stack(labels)
        return padded_images_tensor, labels

    def split_data_train_test(self, data_path: str, split_by: str ='parking_lot'):
        all_dataset = pd.read_csv(data_path)

        unique_parking_lots = all_dataset['parking_lot_id'].unique()
        cleaned_list = []

        logger.info(f"Starting data cleaning with size {len(all_dataset)}")
        for i, parking_lot in enumerate(unique_parking_lots):
            data = all_dataset[all_dataset['parking_lot_id'] == parking_lot]
            cleaned_list.append(clean_data(data['img_path'].to_list()))

        cleaned_list = [elem for elem in cleaned_list[0]]
        all_dataset = all_dataset[all_dataset['img_path'].isin(cleaned_list)]

        logger.info(f"Data clening completed with size : {len(all_dataset)}")

        if split_by == 'size':
            logger.info("Splitting by size")
            train_parking_lot, val_parking_lot = train_test_split(all_dataset, test_size=0.2, random_state=42)
            logger.info(f"Training parking lots of size : {len(train_parking_lot)}")
            logger.info(f"Validation parking lots of size : {len(val_parking_lot)}")
        elif split_by == 'parking_lot':
            logger.info("Splitting by parking lot id")
            logger.info(f"Total number of parking lots found in the dataset: {len(unique_parking_lots)}")
            train_parking_lot, val_parking_lot = train_test_split(all_dataset['parking_lot_id'].unique(),
                                                                  test_size=0.2, random_state=42)
            logger.info(f"Training parking lots of size : {len(train_parking_lot)}")
            logger.info(f"Validation parking lots of size : {len(val_parking_lot)}")
        else:
            raise ValueError("Split by should be either size or parking_lot")
        return train_parking_lot, val_parking_lot
