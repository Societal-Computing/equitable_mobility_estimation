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
