import argparse
import os
import sys

from pathlib import Path
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch
import copy
import pandas as pd
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
import cProfile
import pstats

from utils import *
from models import get_model, optimize_temperature, evaluate_with_temperature_scaling
from base_trainer import EarlyStopping, BaseTrainer
from torch.profiler import tensorboard_trace_handler, profile
from sklearn.model_selection import train_test_split
import random
from typing import Dict
from logger import logger
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler(device=device)

class PlanetRankerDataset(Dataset):
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
        anchor_image_image_path = row['anchor_image']
        anchor_image_pair_image_path = row['anchor_image_pair']

        with rasterio.open(anchor_image_image_path, 'r') as src:
            anchor_img_data = src.read()
            
        with rasterio.open(anchor_image_pair_image_path, 'r') as src:
            anchor_pair_img_data = src.read()

        label = row['label']

        label = torch.tensor(label).long()

        if self.preprocess_type == 'derivatives':
            anchor_img_data = convert_to_image_derivatives(anchor_img_data)
            anchor_pair_img_data = convert_to_image_derivatives(anchor_pair_img_data)
        elif self.preprocess_type == 'rgb':
            anchor_img_data = convert_to_rgb(anchor_img_data)
            anchor_pair_img_data = convert_to_rgb(anchor_pair_img_data)
            #im = self.stacking_rgb_images(img_data) #testing this feature against convert_to_rgb, doesn't work input supports unit8 only
        else:
            raise ValueError(
                "Invalid preprocess type, choose from ['derivatives', 'rgb']")

        if self.transform:
            anchor_img_data_tensor = self.transform(anchor_img_data)
            anchor_pair_img_data_tensor = self.transform(anchor_pair_img_data)
        else:
            anchor_img_data_tensor = torch.as_tensor(anchor_img_data)
            anchor_pair_img_data_tensor = torch.as_tensor(anchor_pair_img_data)

        return anchor_img_data_tensor, anchor_pair_img_data_tensor, label

class PairWiseTrainer(BaseTrainer):
    def __init__(self, config_path: str, early_stopping: bool) -> None:
        config = load_config(config_path)
        self.config = config
        self.model_config = config['model_config']
        self.train_config = config['train_config']
        self.train_data_path = config['data_config']['train_data']['data_path']
        self.test_data_path = config['data_config']['test_data']['data_path']
        self.train_augementations = config['data_config']['train_data']['augmentations']
        self.val_augmentations = config['data_config']['test_data']['augmentations']
        self.is_early_stopping = early_stopping
        self.preprocess_type = config['train_config']['preprocess_type']

        self.__set_seed__(self.train_config['seed'])

        if 'post_temp_scaling' in config and config['post_temp_scaling'] == 'yes':
            self.post_temp_scaling = config['post_temp_scaling']
        else:
            self.post_temp_scaling = False

    @staticmethod
    def __set_seed__(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Seed set to {seed}")

    @staticmethod
    def get_mean_std(dataloader):
        mean = 0.0
        std = 0.0
        nb_samples = 0.0

        for data, _ in dataloader:
            data = data.view(data.size(0), data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += data.size(0)

        mean /= nb_samples
        std /= nb_samples

        return mean, std

    def get_top_bottom_25_percent(self, test_data):
        # get top 25% and bottom 25% of test data based on parking lot name
        unique_parking_lots = test_data['parking_lot_name'].unique()
        top_25_percent = []
        bottom_25_percent = []
        for parking_lot in unique_parking_lots:
            sel_parking_lot = test_data[test_data['parking_lot_name'] == parking_lot].copy()
            sel_parking_lot = sel_parking_lot.sort_values(by='probs', ascending=False).reset_index(drop=True)
            top_25 = sel_parking_lot.loc[:int(0.25 * sel_parking_lot.shape[0])]
            bottom_25 = sel_parking_lot.loc[int(0.75 * sel_parking_lot.shape[0]):]
            top_25.loc[:, 'label'] = 1
            bottom_25.loc[:, 'label'] = 0
            top_25_percent.append(top_25)
            bottom_25_percent.append(bottom_25)
        top_25_percent = pd.concat(top_25_percent)
        bottom_25_percent = pd.concat(bottom_25_percent)
        return top_25_percent, bottom_25_percent

    def plot_occupancy_predictions(self, bootstrapped_predictions_df, city, plot_save_path, data_type='filtering'):
        bootstrapped_pred_sunday = bootstrapped_predictions_df[bootstrapped_predictions_df['day'] == 'sunday']
        bootstrapped_pred_saturday = bootstrapped_predictions_df[bootstrapped_predictions_df['day'] == 'saturday']

        plt.figure(figsize=(30, 20))
        plt.bar(bootstrapped_pred_sunday['parking_lot_name'], bootstrapped_pred_sunday['probs'], color='blue',
                alpha=0.4, label='Sunday Occupancy')
        plt.bar(bootstrapped_pred_saturday['parking_lot_name'], bootstrapped_pred_saturday['probs'], color='red',
                alpha=0.4, label='Saturday Occupancy')
        plt.plot(bootstrapped_pred_sunday['parking_lot_name'], bootstrapped_pred_sunday['probs'], color='blue',
                 alpha=0.7, linestyle='-', marker='o')
        plt.plot(bootstrapped_pred_saturday['parking_lot_name'], bootstrapped_pred_saturday['probs'], color='red',
                 alpha=0.7, linestyle='-', marker='o')

        plt.xlabel("Parking Lot Name", fontsize=12)
        plt.ylabel("Predicted Probability (Occupancy)", fontsize=12)
        plt.title(f"{city}: Highres2Planet Occupancy Prediction", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{plot_save_path}/{city}_{data_type}_occupancy_plot.png")
        plt.close()
    
    @staticmethod
    def get_pair_wise_dataloader(train_data_path: str ,val_data_path: str,train_augmentations: Dict,test_augmentations: Dict, preprocess_type: str, batch_size: int):
        train_transform = get_transforms(train_augmentations)
        test_transform = get_transforms(test_augmentations)

        if isinstance(train_data_path, str):
            train_df = pd.read_csv(train_data_path)
        elif isinstance(train_data_path, pd.DataFrame):
            train_df = train_data_path.copy()
        else:
            raise ValueError("train_data_path should be either a string or a dataframe")


        if isinstance(val_data_path, str):
            test_df = pd.read_csv(val_data_path)
        elif isinstance(val_data_path, pd.DataFrame):
            test_df = val_data_path.copy()
        else:
            raise ValueError("val_data_path should be either a string or a dataframe")

        # all_data = pd.concat([train_df,test_df])
        # unique_lots = all_data['parking_lot_name'].unique()
        # train_lots, test_lots = train_test_split(unique_lots, test_size=0.2, random_state=86)
        #
        # train_df = all_data[all_data['parking_lot_name'].isin(train_lots)]
        # test_df = all_data[all_data['parking_lot_name'].isin(test_lots)]

        #test with a few samples
        train_df = train_df.sample(5000) # 382692 -> 12hr approx
        test_df = test_df.sample(500, random_state=42)

        train_dataset = PlanetRankerDataset(train_df, transform=train_transform, preprocess_type=preprocess_type)
        val_dataset = PlanetRankerDataset(test_df, transform=test_transform, preprocess_type=preprocess_type)

        print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, test_loader, test_df
    def epoch_run(self, model, criterion, optimizer, train_loader, test_loader, scheduler, num_epochs,
                  save_path):
        epoch_train_loss = []
        epoch_test_loss = []
        epoch_train_true = []
        epoch_train_pred = []
        #epoch_test_true = []
        epoch_test_pred = []

        best_loss = np.inf

        early_stopping = EarlyStopping(patience=10, save_path=save_path, model_name=self.model_config['model_name'],
                                       verbose=True)

        for epoch in tqdm(range(1, num_epochs + 1)):
            logger.info(f'Epoch {epoch}/{num_epochs}')

            model.train()
            train_loss = torch.tensor(0.0, device=device)
            total = 0
            all_probs = []
            all_labels = []
            correct = 0

            for anchor_image, anchor_image_pair, target in train_loader:
                anchor_image = anchor_image.to(device)
                anchor_image_pair = anchor_image_pair.to(device)
                target = target.to(device)
                target = target.unsqueeze(1).float()
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    train_pred = model(anchor_image, anchor_image_pair)

                    loss = criterion(train_pred, target)
                    _, predicted = torch.max(train_pred, 1)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                train_loss += loss

                total += target.size(0)
                all_probs.extend(train_pred.detach().cpu().numpy())
                all_labels.extend(target.detach().cpu().numpy())

            train_loss = train_loss.item()
            train_loss /= len(train_loader)
            train_acc = correct / total
            scheduler.step()
            logger.info(f"Epoch {epoch} : Train Loss {train_loss}")

            model.eval()
            test_loss = torch.tensor(0.0, device=device)
            test_all_probs = []
            test_all_labels = []
            correct = 0
            total = 0
            for anchor_image, anchor_image_pair, target in test_loader:
                anchor_image = anchor_image.to(device)
                anchor_image_pair = anchor_image_pair.to(device)
                target = target.to(device)
                target = target.unsqueeze(1).float()
                with torch.no_grad():
                    output = model(anchor_image, anchor_image_pair)

                loss = criterion(output, target)
                test_loss += loss
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                test_all_probs.extend(output.detach().cpu().numpy())
                test_all_labels.extend(target.detach().cpu().numpy())

            test_loss = test_loss.item()
            test_loss /= len(test_loader)
            test_all_probs = np.array(test_all_probs)
            test_all_labels = np.array(test_all_labels)
            logger.info(f"Epoch {epoch} : Test Loss {test_loss}")

            epoch_train_loss.extend([train_loss])
            epoch_test_loss.extend([test_loss])
            epoch_train_true.extend(all_labels)
            epoch_train_pred.extend(all_probs)
            #epoch_test_true.append(test_all_labels)
            epoch_test_pred.append(test_all_probs)

            if best_loss > test_loss:
                best_loss = test_loss
                final_model = model

            if self.is_early_stopping:
                early_stopping(test_loss, final_model)
                final_epoch = epoch
                if early_stopping.early_stop:
                    logger.info(f"Early stopping at Epoch {epoch}...............")
                    break

        if not self.is_early_stopping:
            final_epoch = num_epochs

        return final_model, final_epoch, epoch_train_loss, epoch_test_loss, epoch_train_true, epoch_train_pred, epoch_test_pred


    def train(self):
        model_config = self.model_config
        model = get_model(model_config)
        experiment_dir = generate_experiment_name(f"patch_{model_config['model_name']}")

        model = model.to(device)

        lr = self.train_config['lr']
        optimizer = getattr(optim, self.train_config['optimizer'])(model.parameters(), lr=lr)
        lr_scheduler = getattr(optim.lr_scheduler, self.train_config['scheduler']['scheduler_name'])(optimizer,
                                                                                                     milestones=
                                                                                                     self.train_config[
                                                                                                         'scheduler'][
                                                                                                         'milestones'],
                                                                                                     gamma=
                                                                                                     self.train_config[
                                                                                                         'scheduler'][
                                                                                                         'gamma'])

        criterion = getattr(nn, self.train_config['criterion'])()

        train_loader, test_loader, test_df = self.get_pair_wise_dataloader(self.train_data_path, self.test_data_path,
                                                        self.train_augementations, self.val_augmentations,
                                                        self.preprocess_type, self.train_config['batch_size'])

        final_model, final_epoch, epoch_train_loss, epoch_test_loss, epoch_train_true, epoch_train_pred, epoch_test_pred = self.epoch_run(
            model, criterion, optimizer, train_loader, test_loader, lr_scheduler,
            self.train_config['num_epochs'], experiment_dir)

        torch.save(final_model.state_dict(), experiment_dir.joinpath(f"{self.model_config['model_name']}.pt"))
        logger.info(f"Model saved at {experiment_dir}")

        epoch_test_pred_np = np.array(epoch_test_pred)
        epoch_test_pred_np = np.mean(epoch_test_pred_np, axis=0)
        test_df['pred'] = epoch_test_pred_np
        test_df.to_csv(f"{experiment_dir}/test_predictions.csv", index=False)
        plots(final_epoch, epoch_train_loss, epoch_test_loss, epoch_train_true,
              epoch_train_pred, test_df['label'], epoch_test_pred, experiment_dir, 'classification')

        if self.post_temp_scaling:
            temperature_scaling, optimized_model = optimize_temperature(final_model, test_loader, criterion,
                                                                        self.model_config['model_name'])
            logger.info(f"Temperature scaling factor: {temperature_scaling}")
            update_config = copy.deepcopy(self.config)
            update_config['model_config']['temperature_scaling'] = temperature_scaling
            save_inference_config(update_config, experiment_dir, train_type='single')

            torch.save(final_model.state_dict(), experiment_dir.joinpath(f"{self.model_config['model_name']}.pt"))
            logger.info(f"Model saved after temperature scaling {experiment_dir}")

            evaluate_with_temperature_scaling(final_model, optimized_model, test_loader, temperature_scaling,
                                              self.model_config['model_name'])

        else:
            logger.info(f"Model saved with no temp scaling at {experiment_dir}")
            save_inference_config(self.config, experiment_dir, train_type='single')

        return experiment_dir

    def load_model(self, model_config, model_weight):
        model = get_model(model_config)
        logger.info(f"Loading model from {model_weight}")
        model.load_state_dict(torch.load(model_weight))
        return model

    @staticmethod
    def plot_bootstrap_errorbars(bootstrapped_predictions, parking_lots, city, plot_save_path, data_type='filtering'):
        print(bootstrapped_predictions.shape)
        city = "test"
        bootstrapped_predictions = bootstrapped_predictions[:, :, 1]
        for i in range(0, bootstrapped_predictions.shape[1], 50):
            sample = bootstrapped_predictions[:, i:i + 50]
            plt.figure(figsize=(30, 20))
            x_axis = parking_lots[i:i + 50]
            plt.boxplot(sample, showfliers=True)
            plt.xlabel("Test Image Indicies", fontsize=12)
            plt.ylabel("Predicted Probability (Occupancy)", fontsize=12)
            plt.title(f"{city}: Prediction Distribution with Uncertainty Bounds", fontsize=14)
            plt.xticks(ticks=range(1, len(x_axis) + 1), labels=x_axis, rotation=75, ha='right', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{plot_save_path}/{city}_{data_type}_{i}_CI_plot.png")
        plt.close()


if __name__ == "__main__":
    #writer = SummaryWriter()
    #with profile(on_trace_ready=tensorboard_trace_handler("logs")) as prof:
    pr = cProfile.Profile()
    pr.enable()
    parser = argparse.ArgumentParser(
        usage="python src/pair_wise_ranker_trainer.py --config_path configs/pair_wise_config.yaml",
        description="Training file")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    parser.add_argument("--early_stopping", type=str, help="Early stopping", choices=["true", "false"])

    args = parser.parse_args()
    config_path = args.config_path
    early_stopping = args.early_stopping
    bool_mapper = {"true": True, "false": False}
    early_stopping = bool_mapper[early_stopping.lower()]

    trainer = PairWiseTrainer(config_path, early_stopping)
    experiment_dir = trainer.train()

    pr.disable()
    # pr.dump_stats("profile_results_no_mp_no_preloading.prof")

    with open(f"{experiment_dir}/profile_results.txt", "w") as f:
        ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
        ps.print_stats()

    #writer.close()