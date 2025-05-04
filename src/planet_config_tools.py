import datetime
import json
import os
import pathlib
import time
from datetime import datetime, timedelta

import requests
from requests.auth import HTTPBasicAuth

from logger import logger

HEADERS = {'Content-Type': 'application/json'}


def get_block_dates(start_date, end_date, use_block_dates=False):
    if use_block_dates:
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        start_date = start_date[5:]
        end_date = end_date[5:]
        date_ranges = [[f"{year}-{start_date}", f"{year}-{end_date}"] for year in range(start_year, end_year + 1)]
    else:
        date_ranges = [[start_date, end_date]]
    return date_ranges


def place_order(request, auth):
    response = requests.post('https://api.planet.com/compute/ops/orders/v2', data=json.dumps(request), auth=auth,
                             headers=HEADERS)
    order_id = response.json()['id']
    order_url = 'https://api.planet.com/compute/ops/orders/v2' + '/' + order_id
    return order_url


def poll_for_success(order_url, api_key):
    while True:
        r = requests.get(order_url, auth=(api_key, ""))
        response = r.json()
        state = response['state']
        logger.info(state)
        end_states = ['success', 'failed', 'partial']
        if state in end_states:
            break
        time.sleep(10)


def download_results(results, geojson_file, path2save, overwrite=False):
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    logger.info(f'{len(results_urls)} items to download')

    cluster_num = os.path.splitext(geojson_file)[0]

    for url, name in zip(results_urls, results_names):
        path = pathlib.Path(os.path.join(path2save, cluster_num, name))

        if overwrite or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
            r = requests.get(url, allow_redirects=True)
            open(path, 'wb').write(r.content)
        else:
            logger.info(f'{path} already exists, skipping {name}')


def find_day(date_string):
    try:
        date = datetime.strptime(date_string, "%Y-%m-%d")
        day = date.strftime("%A")
        return day
    except ValueError:
        return "Invalid date format. Please provide the date in YYYY-MM-DD format."


def iterate_over_selected_dates(start_date, end_date, selected_days="all"):
    '''Returns a list of dates of particular days between the start_date and end_date.'''
    if selected_days == "all":
        selected_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    selected_date_list = []

    current_date = start_date
    while current_date <= end_date:
        if find_day(current_date.isoformat()[:10]) in selected_days:
            selected_date_list.append(current_date.isoformat()[:10])
        current_date += timedelta(days=1)

    return selected_date_list


def search_payload(item_type, product_bundle_type, quality_type, geom, start_date, end_date, cloud_cover):
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geom
    }
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": start_date,
            "lte": end_date
        }
    }
    cloud_cover_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
            "lt": cloud_cover
        }
    }
    product_bundle_filter = {
        "type": "AssetFilter",
        "config": [product_bundle_type]
    }
    quality_filter = {
        "type": "StringInFilter",
        "field_name": "quality_category",
        "config": [quality_type]
    }
    combined_filter = {
        "type": "AndFilter",
        "config": [geometry_filter, date_range_filter, cloud_cover_filter, product_bundle_filter, quality_filter]
    }
    search_request = {
        "item_types": [item_type],
        "filter": combined_filter
    }
    return search_request


def yield_features(url, auth, payload):
    page = requests.post(url, auth=auth, data=json.dumps(payload), headers=HEADERS)
    for feature in page.json()['features']:
        yield feature

    while True:
        next_url = page.json()['_links'].get('_next')
        if not next_url:
            break
        page = requests.get(next_url, auth=auth)

        for feature in page.json()['features']:
            yield feature


def ft_iterate(api_key, item_type, product_bundle_type, quality_type, geom, start_date, end_date, cloud_cover):
    
    search_json = search_payload(item_type, product_bundle_type, quality_type, geom, start_date, end_date, cloud_cover)
    all_features = list(
        yield_features('https://api.planet.com/data/v1/quick-search', HTTPBasicAuth(api_key, ''), search_json))
    image_ids = [x['id'] for x in all_features]
    return image_ids


def order_payload(geojson_file, geom, image_ids, item_type, order_type, product_bundle="analytic_8b_sr_udm2"):
    payload = {
        "name": os.path.splitext(geojson_file)[0],
        "order_type": order_type,
        "products": [
            {
                "item_ids": image_ids,
                "item_type": item_type,
                "product_bundle": product_bundle
            }
        ],
        "tools": [
            {
                "clip": {
                    "aoi": geom
                }
            }
        ]
    }

    return payload
