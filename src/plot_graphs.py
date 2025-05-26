from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

"""
Usage:

python src/plot_graphs.py \
    --input_dir downloads/results/ \
    --activation_aggregation last \
    --prompt_name empty
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, default="downloads/results/")
    parser.add_argument("--model_checkpoint", type=str, required=True, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--activation_aggregation", type=str, default="last")
    parser.add_argument("--prompt_name", type=str, default="name")
    return parser.parse_args()


def main(args):
    model_name = args.model_checkpoint.split("/")[-1]
    path = Path(args.input_dir) / model_name / args.prompt_name
    csv_files = sorted(path.glob(f"*.csv"))

    plt.figure(figsize=(8, 5))

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "cyan",
        "lime",
        "teal",
        "navy",
        "maroon",
        "olive",
    ]

    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        city = csv_file.stem.split(".")[-3].replace("places_", "")
        city = " ".join(city.split("_")).title()
        plt.plot(df["layer"] / max(df["layer"]), df["r2"], linestyle="-", label=city, color=colors[idx])

    plt.xlabel("Model Depth")
    plt.ylabel("Places test R²")
    plt.title(f"{model_name}, Prompt type: {args.prompt_name}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(-0.5, 1.0)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fig_save_name = f"{model_name}.{args.activation_aggregation}.{args.prompt_name}.png"
    plt.savefig(path / fig_save_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
