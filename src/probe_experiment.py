# Modified from: https://github.com/wesg52/world-models/blob/main/probe_experiment.py

from argparse import ArgumentParser
import warnings
import os

from tqdm import tqdm
from sklearn.linear_model import Ridge, RidgeCV

import numpy as np
import pandas as pd
import torch

from datasets import load_from_disk
from probe_evaluation import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

MODEL_N_LAYERS = {
    "Llama-2-7b-hf": 32,
    "Llama-3.2-1B": 16,
    "Llama-3.1-8B": 32,
    "aya-expanse-8b": 32,
}

RIDGE_ALPHAS = {
    "Llama-2-7b-hf": np.logspace(0.8, 4.1, 12),
    "Llama-3.2-1B": np.logspace(0.8, 4.1, 12),
    "Llama-3.1-8B": np.logspace(0.8, 4.1, 12),
    "aya-expanse-8b": np.logspace(0.8, 4.1, 12),
}

"""
Usage:

python src/probe_experiment.py \
    --model_checkpoint meta-llama/Llama-3.1-8B \
    --dataset_save_path places_dataset \
    --activation_save_path activation_datasets \
    --activation_aggregation last \
    --prompt_name empty \
    --output_dir probing_results
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_save_path", type=str, required=True, default="places_dataset")
    parser.add_argument("--activation_save_path", type=str, required=True, default="activation_datasets")
    parser.add_argument("--activation_aggregation", type=str, default="last")
    parser.add_argument("--prompt_name", type=str, default="empty")
    parser.add_argument("--model_checkpoint", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output_dir", type=str, default="probing_results")
    return parser.parse_args()


def place_probe_experiment(activations, target, is_test, probe=None, is_lat_lon=True):
    train_activations = activations[~is_test]
    train_target = target[~is_test]

    test_activations = activations[is_test]
    test_target = target[is_test]

    norm_train_target = (train_target - train_target.mean(axis=0)) / train_target.std(axis=0)

    if probe is None:
        probe = Ridge(alpha=activations.shape[1])

    probe.fit(train_activations, norm_train_target)

    train_pred = probe.predict(train_activations)
    test_pred = probe.predict(test_activations)

    train_pred_unnorm = train_pred * train_target.std(axis=0) + train_target.mean(axis=0)
    test_pred_unnorm = test_pred * train_target.std(axis=0) + train_target.mean(axis=0)

    projection = probe.predict(activations) * train_target.std(axis=0) + train_target.mean(axis=0)

    train_scores = score_place_probe(train_target, train_pred_unnorm, use_haversine=is_lat_lon)
    test_scores = score_place_probe(test_target, test_pred_unnorm, use_haversine=is_lat_lon)

    scores = {
        **{("train", k): v for k, v in train_scores.items()},
        **{("test", k): v for k, v in test_scores.items()},
    }

    error_matrix = compute_proximity_error_matrix(target, projection, pairwise_haversine_distance)

    train_error, test_error, combined_error = proximity_scores(error_matrix, is_test)
    scores["train", "prox_error"] = train_error.mean()
    scores["test", "prox_error"] = test_error.mean()

    projection_df = pd.DataFrame(
        {
            "x": projection[:, 0],
            "y": projection[:, 1],
            "is_test": is_test,
            "x_error": projection[:, 0] - target[:, 0],
            "y_error": projection[:, 1] - target[:, 1],
            "prox_error": combined_error,
        }
    )
    return probe, scores, projection_df


def main(args):
    model_name = args.model_checkpoint.split("/")[-1]
    dataset_save_path = os.path.join(args.dataset_save_path, model_name)
    tokenized_dataset = load_from_disk(dataset_save_path)
    entity_df = tokenized_dataset.to_pandas()

    n_layers = MODEL_N_LAYERS[model_name]
    is_test = entity_df.is_test.values

    results = {
        "scores": {},
        "projections": {},
        "probe_directions": {},
        "probe_biases": {},
        "probe_alphas": {},
    }
    for layer in tqdm(range(n_layers)):
        save_name = f"places.{args.activation_aggregation}.{args.prompt_name}.{layer}.pt"
        save_path = os.path.join(args.activation_save_path, model_name, "places", save_name)

        # load data
        activations = torch.load(save_path, weights_only=False).dequantize()

        if activations.isnan().any():
            print("WARNING: nan activations, skipping layer", layer)
            continue

        target = entity_df[["longitude", "latitude"]].values

        probe = RidgeCV(alphas=RIDGE_ALPHAS[model_name], store_cv_values=True)
        probe, scores, projection = place_probe_experiment(activations, target, is_test, probe=probe)

        probe_direction = probe.coef_.T.astype(np.float16)
        probe_alphas = probe.cv_values_.mean(axis=(0, 1))

        results["scores"][layer] = scores
        results["projections"][layer] = projection
        results["probe_directions"][layer] = probe_direction
        results["probe_biases"][layer] = probe.intercept_
        results["probe_alphas"][layer] = probe_alphas

    scores = results["scores"]
    layer_ids = list(scores.keys())
    r2_scores = [scores[layer]["test", "haversine_r2"] for layer in scores]
    results = pd.DataFrame({"layer": layer_ids, "r2": r2_scores})

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_name = f"{model_name}.places.{args.activation_aggregation}.{args.prompt_name}.csv"
    results.to_csv(os.path.join(output_dir, save_name), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
