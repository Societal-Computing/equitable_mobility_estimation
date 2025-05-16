import glob
import random

from itertools import cycle
from typing import Dict, List
import argparse

from planet_config_tools import *
from utils import *

class Planet_Image_Downloader_Block:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.payload_config = self.config.pop("payload_param")
        self.data_config = self.config.pop("data_param")
        api_keys = self.data_config.pop("api_keys")
        random.shuffle(api_keys)
        self.api_keys = api_keys
        self.api_key = self.api_keys[0]
            

    PRODUCT_BUNDLE_TYPES = {
        "ortho_analytic_8b_sr": "analytic_8b_sr_udm2", # for 8Band Analytic Scenes
        "ortho_analytic_4b_sr": "analytic_sr_udm2", # for 4Band Analytic Scenes
        "ortho_visual": "visual" # for RGB Visual Scenes
    }

    def get_geojson_files(self, geojson_folder: str):
        return glob.glob(f"{geojson_folder}/*.geojson") # returns a list of geojson files in the folder

    @staticmethod
    def run_with_iterative_api_keys(current_api_key: str, api_keys: List[str], function, *args, **kwargs):
        api_keys_cycle = cycle(api_keys)
        while api_keys:
            try:
                result = function(*args, **kwargs)
                return result, current_api_key
            except Exception as e:
                logger.info(f"An error occurred with API key {current_api_key}: {e}")
                current_api_key = next(api_keys_cycle, None)
                if current_api_key is None:
                    break
        return None, None

    def main(self):
        payload_config = self.payload_config.copy()
        start_date = payload_config['block_date']['start_date']
        end_date = payload_config['block_date']['end_date']
        item_type = payload_config['item_type']
        product_bundle_type = payload_config['product_bundle_type']
        quality_type = payload_config['quality_type']
        cloud_cover = payload_config['cloud_cover']
        selected_days = payload_config['selected_days']
        order_type = payload_config['order_type']
        geojson_folder = self.data_config['geojson_dir']
        saving_folder = self.data_config['saving_dir']
        api_key = self.api_key
        api_keys = self.api_keys
        product_bundle = self.PRODUCT_BUNDLE_TYPES[product_bundle_type]
        geojson_files = self.get_geojson_files(geojson_folder)
        date_ranges = get_block_dates(start_date, end_date)
        
        list_of_error_geo_json_files = []

        for i, geojson_file in enumerate(geojson_files):
            logger.info(f"Processing {geojson_file} ...")
            
            try:
                logger.info(f"using api_key: {api_key}")
                results = self.search_result_for_single_geojson(geojson_file, date_ranges, selected_days, api_key,
                                                                item_type, product_bundle_type, quality_type,
                                                                cloud_cover, order_type, product_bundle)

            except Exception as e:
                logger.info(f"An error occurred: {e}, changing api key ...")
                previous_api_key = api_key
                api_keys.remove(api_key)
                random.shuffle(api_keys)
                api_key = self.cyle_api_key(api_keys, api_key)
                logger.info(f"Api_key changed :{previous_api_key} -> {api_key}")
                list_of_error_geo_json_files.append(geojson_file)
                logger.info(
                    f"File failed to download due to api quota exceeded, has been added to files to retry, current size is {len(list_of_error_geo_json_files)}")
                time.sleep(10)
                continue

            download_results(results, geojson_file, saving_folder)
            logger.info(f"Downloaded images saved in {saving_folder}")
        logger.info(f" {len(list_of_error_geo_json_files)} files failed to download we are retrying...")

        if len(list_of_error_geo_json_files) == 0:
            logger.info("All files downloaded successfully")
            return

        logger.info(f"Retrying failed files of size {len(list_of_error_geo_json_files)} due to api key issues.")
        for geojson_file in list_of_error_geo_json_files:
            try:
                logger.info(f"api_key: {api_key}")
                results = self.search_result_for_single_geojson(geojson_file, date_ranges, selected_days, api_key,
                                                                item_type, product_bundle_type, quality_type,
                                                                cloud_cover, order_type, product_bundle)

            except Exception as e:
                previous_api_key = api_key
                api_keys.remove(api_key)
                random.shuffle(api_keys)
                api_key = self.cyle_api_key(api_keys, api_key)
                logger.info(f"An error occurred: {e}, changing api key...")
                list_of_error_geo_json_files.append(geojson_file)
                logger.info(f"api_key changed :{previous_api_key} -> {api_key}")
                time.sleep(10)
                continue

            download_results(results, geojson_file, saving_folder)
            logger.info(f"Downloaded images saved in {saving_folder}")

    def cyle_api_key(self, api_keys, api_key):
        api_keys_cycle = cycle(api_keys)
        current_api_key = next(api_keys_cycle, None)
        while api_key == current_api_key:
            current_api_key = next(api_keys_cycle, None)
        return current_api_key

    def search_result_for_single_geojson(self, geojson_file, date_ranges, selected_days, api_key, item_type,
                                         product_bundle_type, quality_type, cloud_cover, order_type,
                                         product_bundle):
        geom = read_geom(geojson_file)
        all_image_ids = []

        for date_range in date_ranges:
            start_date = date_range[0]
            end_date = date_range[1]
            logger.info(f"start_date: {start_date}.... end_date: {end_date}")

            selected_dates = iterate_over_selected_dates(start_date, end_date, selected_days)

            for date in selected_dates:
                start_date_local = date + "T00:00:00.000Z"
                end_date_local = date + "T23:59:59.999Z"

                image_ids = ft_iterate(api_key, item_type, product_bundle_type, quality_type, geom, start_date_local,
                                       end_date_local, cloud_cover)

                if image_ids:
                    all_image_ids.extend(image_ids)

        logger.info(f"Total image ids:{len(all_image_ids)}")

        order_request = order_payload(geojson_file, geom, all_image_ids, item_type, order_type, product_bundle)

        order_url = place_order(order_request, auth=(api_key, ""))

        poll_for_success(order_url, api_key)

        r = requests.get(order_url, auth=(api_key, ""))
        response = r.json()
        results = response['_links']['results']
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download planet assets",
                                     usage="python src/planet_imagery_downloader.py --config_path configs/downloader_config.yaml")
    
    parser.add_argument('--config_path', type=str, help='path to config file')
    args = parser.parse_args()
    planet_obj = Planet_Image_Downloader_Block(args.config_path)
    planet_obj.main()
