# Modified from: https://github.com/wesg52/world-models/blob/main/make_prompt_datasets.py

from argparse import ArgumentParser
import os

from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import torch

"""
Usage:

python src/make_prompt_dataset.py \
    --entity_file downloads/nyc_place.csv \
    --model_checkpoint meta-llama/Llama-3.2-1B \
    --dataset_save_path downloads/nyc_place_dataset
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--entity_file", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset_save_path", type=str, required=True)
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


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    entity_df = pd.read_csv(args.entity_file)
    entity_list = list(entity_df["name"].values)
    entity_list = normalize_location_names(entity_list)

    token_ids = tokenizer.batch_encode_plus(
        entity_list,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    # add bos token
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
        }
    )
    tokenized_dataset.set_format(type="torch")

    model_name = args.model_checkpoint.split("/")[-1]
    dataset_save_path = os.path.join(args.dataset_save_path, model_name)
    os.makedirs(dataset_save_path, exist_ok=True)

    tokenized_dataset.save_to_disk(dataset_save_path)
    print(f"Saved prompt dataset to {dataset_save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
