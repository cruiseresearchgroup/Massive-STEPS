# Modified from: https://github.com/wesg52/world-models/blob/main/make_prompt_datasets.py

from argparse import ArgumentParser
import os

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from transformers import AutoTokenizer
from datasets import Dataset
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
    parser.add_argument("--remove_outliers", action="store_true")
    return parser.parse_args()


def normalize_location_names(locations):
    stop_words = {"AND", "OR", "OF", "THE", "A", "AT", "&", "IN", "TO"}
    abbv_words = {
        "FDNY",
        "NYCT",
        "YMCA",
        "LGA",
        "US",
        "NYC",
        "PS",
        "IS",
        "NYS",
        "UN",
        "NY",
        "EMS",
        "JCC",
        "NYU",
        "CC",
        "NYPD",
        "NYPA",
        "DHS",
    }
    normalized_names = []

    for location in locations:
        words = location.split()
        normalized_name = []
        for i, word in enumerate(words):
            if word.strip() in stop_words:
                normalized_name.append(word.lower())
            elif word.strip() in abbv_words:
                normalized_name.append(word)
            else:
                normalized_name.append(word.lower().capitalize())
        normalized_names.append(" ".join(normalized_name))

    return normalized_names


def normalize_city_names(cities):
    def _normalize(city):
        return " ".join([word.capitalize() for word in city.split("_")])

    return [_normalize(city) for city in cities]


def make_places_prompt_dataset(tokenizer, entity_df):
    train_df, test_df = train_test_split(entity_df, test_size=0.2, random_state=42, stratify=entity_df["city"])

    def _get_list(train_df, test_df, col):
        return train_df[col].to_list() + test_df[col].to_list()

    entity_list = _get_list(train_df, test_df, "name")
    address_list = _get_list(train_df, test_df, "address")
    city_list = _get_list(train_df, test_df, "city")
    latitude_list = _get_list(train_df, test_df, "latitude")
    longitude_list = _get_list(train_df, test_df, "longitude")
    is_test = [0] * len(train_df) + [1] * len(test_df)

    city_list = normalize_city_names(city_list)
    entity_list = [f"{address}, {city.capitalize()}" for address, city in zip(address_list, city_list)]
    entity_list = normalize_location_names(entity_list)

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


def main(args):
    cities = [
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

    if args.city is not None:
        city_path = os.path.join(args.data_path, args.city)
        entity_df = pd.read_csv(os.path.join(city_path, f"{args.city}_checkins.csv"))
        entity_df = (
            entity_df.groupby(["name", "address", "latitude", "longitude"]).size().reset_index(name="count").dropna()
        )
        entity_df = entity_df[entity_df["count"] >= 10]  # only keep POIs with at least 10 check-ins
        entity_df["city"] = args.city
    else:
        all_df = []
        for city in cities:
            city_path = os.path.join(args.data_path, city)
            df = pd.read_csv(os.path.join(city_path, f"{city}_checkins.csv"))
            df = df.groupby(["name", "address", "latitude", "longitude"]).size().reset_index(name="count").dropna()
            df = df[df["count"] >= 10]  # only keep POIs with at least 10 check-ins
            df["city"] = city
            all_df.append(df)

        entity_df = pd.concat(all_df, ignore_index=True)

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
