# 🚗 A Weak Supervision Learning Approach Towards an Equitable Mobility Estimation

This repository contains the code to reproduce the results of our workshop paper titled _"A Weak Supervision Learning Approach Towards an Equitable Mobility Estimation"_ accepted at **ICWSM 2025**, workshop _Data for the Wellbeing of the Most Vulnerable_.

### 📝 Abstract
The scarcity and high cost of labeled high-resolution imagery have long challenged remote sensing applications, particularly in low-income regions where high-resolution data are scarce. In this study, we propose a weak supervision framework that estimates parking lot occupancy using 3m resolution satellite imagery. By leveraging coarse temporal labels—based on the assumption that parking lots of major supermarkets and hardware stores in Germany are typically full on Saturdays and empty on Sundays — we train a pairwise comparison model that achieves an AUC of 0.92 on large parking lots. The proposed approach minimizes the reliance on expensive high-resolution images and holds promise for scalable urban mobility analysis. Moreover, the method can be adapted to assess transit patterns and resource allocation in vulnerable communities, providing a data-driven basis to improve the well-being of those most in need.

### 📦 Project Structure

```cmd
./
├── README.md*
├── configs/
│   ├── downloader_config.yaml*
│   └── pair_wise_config.yaml*
├── notebooks/
│   ├── 01_parking_lot_extraction.ipynb*
│   └── 02_pairwise_ranking_applications.ipynb*
├── requirements.txt*
├── results/
├── src/
   ├── __init__.py*
   ├── base_trainer.py*
   ├── data_preprocessing.py*
   ├── logger.py*
   ├── models.py*
   ├── pair_wise_ranker_trainer.py*
   ├── planet_config_tools.py*
   ├── planet_imagery_downloader.py*
   └── utils.py*

```
### 🔧 Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 🛠️ Install Dependencies
```python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Data Preparation
#### 🗺️ Step1: Extraction of Parking lot polygons
Use OSM data to extract parking lot polygons:
Run: `notebooks/01_parking_lot_extraction.ipynb`

#### 🛰️ Planet Image Download
Use the following command to download Planet 3m resolution satellite images:
```bash
python src/planet_imagery_downloader.py --config_path configs/downloader_config.yaml
```

#### 🧹 Data Preprocessing
Clean and mask imagery based on polygons:

```bash
python data_preprocessing.py --geojson_base_dir dataset/parking_lots --masked_images_root dataset/Masked_Images 
--save_path dataset --split_data True
```
`geojson_base_dir` a directory which contains all polygon files extracetd from OSM

`masked_images_root` a directory which contains all individual parking lot images 

`save_path` the directory to save the output csv of the filtered images

`split-data` use this option to split data into train/test

### 🧠 Training of pairwise model
To train the pairwise ranking model with your prepared data:

```bash
python src/pair_wise_ranker_trainer.py --config_path configs/pair_wise_config.yaml
```

### 🌍 Application: Sudan
The evaluation application task is found in `notebooks\02_pairwise_ranking_applications.ipynb`

#### 📖 Cite our paper:
```bibtex
@article{aidoo2025weak,
  title={A Weak Supervision Learning Approach Towards an Equitable Parking Lot Occupancy Estimation},
  author={Aidoo, Theophilus and Koebe, Till and Maurya, Akansh and Shrestha, Hewan and Weber, Ingmar},
  journal={arXiv preprint arXiv:2505.04229},
  year={2025}
}
```