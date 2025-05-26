# Modified from: https://github.com/wesg52/world-models/blob/main/make_prompt_datasets.py

from argparse import ArgumentParser
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from tqdm.contrib.concurrent import thread_map
from shapely.geometry import Point, shape
from transformers import AutoTokenizer
from datasets import Dataset
import geopandas as gpd
import pandas as pd
import numpy as np
import torch


"""
Usage:

python make_prompt_dataset.py \
    --data_path data \
    --model_checkpoint meta-llama/Llama-3.1-8B \
    --dataset_save_path places_dataset
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset_save_path", type=str, required=True)
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--prompt_name", type=str, default="name")
    parser.add_argument("--min_checkins", type=int, default=2)
    parser.add_argument("--remove_outliers", action="store_true")
    return parser.parse_args()


def make_prompt(name, address, city, country, prompt_name):
    code2name = {
        "AU": "Australia",
        "BR": "Brazil",
        "CN": "China",
        "ID": "Indonesia",
        "JP": "Japan",
        "KW": "Kuwait",
        "MY": "Malaysia",
        "RU": "Russia",
        "TR": "Turkey",
        "US": "United States",
    }

    if prompt_name == "name":
        prompt = name
    elif prompt_name == "address":
        prompt = address
    elif prompt_name == "name_address":
        prompt = f"{name}, {address}"
    elif prompt_name == "name_address_city":
        prompt = f"{name}, {address}, {city}"
    elif prompt_name == "name_address_city_country":
        prompt = f"{name}, {address}, {city}, {code2name[country]}"
    elif prompt_name == "address_city":
        prompt = f"{address}, {city}"
    else:
        raise ValueError(f"Unknown prompt name: {prompt_name}")

    return prompt


def make_places_prompt_dataset(tokenizer, entity_df):
    train_df, test_df = train_test_split(entity_df, test_size=0.2, random_state=42, stratify=entity_df["city"])

    def _get_list(train_df, test_df, col):
        return train_df[col].to_list() + test_df[col].to_list()

    name_list = _get_list(train_df, test_df, "name")
    address_list = _get_list(train_df, test_df, "address")
    city_list = _get_list(train_df, test_df, "venue_city")
    country_list = _get_list(train_df, test_df, "venue_country")
    latitude_list = _get_list(train_df, test_df, "latitude")
    longitude_list = _get_list(train_df, test_df, "longitude")
    is_test = [0] * len(train_df) + [1] * len(test_df)

    entity_list = [
        make_prompt(name, address, city, country, args.prompt_name)
        for name, address, city, country in zip(name_list, address_list, city_list, country_list)
    ]

    token_ids = tokenizer.batch_encode_plus(
        entity_list,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    token_ids = torch.cat(
        [torch.ones(token_ids.shape[0], 1, dtype=torch.long) * tokenizer.bos_token_id, token_ids], dim=1
    )

    prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
    entity_mask = torch.ones_like(token_ids, dtype=torch.bool)
    entity_mask[:, prompt_tokens] = False
    entity_mask[token_ids == tokenizer.pad_token_id] = False

    tokenized_dataset = Dataset.from_dict(
        {
            "entity": entity_list,
            "input_ids": token_ids.tolist(),
            "entity_mask": entity_mask.tolist(),
            "city": city_list,
            "address": address_list,
            "latitude": latitude_list,
            "longitude": longitude_list,
            "is_test": is_test,
        }
    )
    tokenized_dataset.set_format(type="torch")
    return tokenized_dataset


def remove_outliers(entity_df, outliers_percentile=95):
    coords = entity_df[["latitude", "longitude"]].to_numpy()
    mean = np.mean(coords, axis=0)
    cov = np.cov(coords.T)
    inv_cov = np.linalg.inv(cov)

    entity_df["mahalanobis"] = [mahalanobis(x, mean, inv_cov) for x in coords]
    entity_df = entity_df[entity_df["mahalanobis"] < np.percentile(entity_df["mahalanobis"], outliers_percentile)]
    return entity_df


def is_in_polygon(lon, lat, polygon):
    point = Point(float(lon), float(lat))
    return polygon.intersects(point)


def filter_places_by_polygon(entity_df, city_geo_json_file):
    gdf = gpd.read_file(city_geo_json_file)
    gdf = gdf[(gdf["type"] == "boundary") | (gdf["type"] == "multipolygon")]
    city_polygon = shape(gdf.geometry.values[0])

    checkins_coordinates = entity_df[["longitude", "latitude"]].to_numpy()
    entity_df["in_polygon"] = thread_map(lambda c: is_in_polygon(c[0], c[1], city_polygon), checkins_coordinates)
    entity_df = entity_df[entity_df["in_polygon"]]
    return entity_df


def load_places_df(data_path, city, min_checkins):
    city_path = os.path.join(data_path, city)
    df = pd.read_csv(os.path.join(city_path, f"{city}_checkins.csv"))
    columns = ["name", "address", "latitude", "longitude", "venue_city", "venue_country"]
    df = df.groupby(columns).size().reset_index(name="num_checkins").dropna()
    df = df[df["num_checkins"] >= min_checkins]
    df["city"] = city
    return df


def get_city_geo_json_file(data_path, city):
    city_geo_json_file = list(Path(data_path).glob(f"{city}/{city}_*_overpass.geojson"))
    assert len(city_geo_json_file) == 1, f"GeoJSON file for {city} not found."
    return city_geo_json_file[0]


def main(args):
    all_cities = [
        "beijing",
        "istanbul",
        "jakarta",
        "kuwait_city",
        "melbourne",
        "moscow",
        "new_york",
        "petaling_jaya",
        "sao_paulo",
        "shanghai",
        "sydney",
        "tokyo",
    ]

    cities = all_cities if args.city is None else [args.city]
    all_df = [load_places_df(args.data_path, city, min_checkins=args.min_checkins) for city in cities]
    entity_df = pd.concat(all_df, ignore_index=True)
    entity_df = filter_places_by_polygon(entity_df, get_city_geo_json_file(args.data_path, args.city))

    if args.remove_outliers:
        entity_df = remove_outliers(entity_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model_name = args.model_checkpoint.split("/")[-1]
    dataset_save_path = os.path.join(args.dataset_save_path, model_name)
    os.makedirs(dataset_save_path, exist_ok=True)

    tokenized_dataset = make_places_prompt_dataset(tokenizer, entity_df)
    tokenized_dataset.save_to_disk(dataset_save_path)
    print(f"Saved prompt dataset to {dataset_save_path}")
    print(tokenized_dataset[0])


if __name__ == "__main__":
    args = parse_args()
    main(args)
