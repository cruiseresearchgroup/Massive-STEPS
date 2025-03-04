from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

"""
Usage:

python src/plot_graphs.py \
    --input_dir downloads/ \
    --entity_type nyc_place \
    --activation_aggregation last \
    --prompt_name empty
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, default="downloads/")
    parser.add_argument("--entity_type", type=str)
    parser.add_argument("--activation_aggregation", type=str, default="last")
    parser.add_argument("--prompt_name", type=str, default="empty")
    return parser.parse_args()


def main(args):
    path = Path(args.input_dir)
    csv_files = list(path.glob(f"*.{args.entity_type}.{args.activation_aggregation}.{args.prompt_name}.csv"))

    plt.figure(figsize=(8, 5))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        model_name = ".".join(csv_file.stem.split(".")[:-3])
        plt.plot(df["layer"], df["r2"], linestyle="-", label=model_name)

    plt.xlabel("Model Depth")
    plt.ylabel("R² Score")
    plt.title("NYC Places")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    fig_save_name = f"{args.entity_type}.{args.activation_aggregation}.{args.prompt_name}.png"
    plt.savefig(f"downloads/{fig_save_name}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
