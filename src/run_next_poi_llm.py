from argparse import ArgumentParser
from pathlib import Path
import json

from datasets import load_dataset
from tqdm.contrib.concurrent import thread_map
import numpy as np

from prompts import prompt_generator
from utils import convert_to_tuple_records
from llm import Gemini


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/run_fsq.sh#L1
    parser.add_argument("--num_users", type=int, default=200)
    # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/utils.py#L168
    parser.add_argument("--num_historical_stays", type=int, default=15)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def calculate_metrics(results):
    acc_1, acc_5, ndcg_5 = 0, 0, 0
    for result in results:
        predictions, ground_truth = result["prediction"], result["ground_truth"]
        acc_1 += 1 if len(predictions) > 0 and predictions[0] == ground_truth else 0
        acc_5 += 1 if ground_truth in predictions else 0
        ndcg_5 += (1 / (predictions.index(ground_truth) + 1)) if ground_truth in predictions else 0

    return {"acc_1": acc_1 / len(results), "acc_5": acc_5 / len(results), "ndcg_5": ndcg_5 / len(results)}


def main(args):
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = args.model_name.split("/")[-1]
    output_dir = args.output_dir / dataset_name / model_name / args.prompt_type
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name)
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    users = sorted(test_df["user_id"].unique())
    users = np.random.RandomState(args.seed).choice(users, min(args.num_users, len(users)), replace=False)

    if "gemini" in args.model_name:
        llm = Gemini(model=args.model_name)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    def generate_prediction(user_id):
        output_file_path = output_dir / f"user_{user_id}.json"

        train_df_user = train_df[train_df["user_id"] == user_id]
        test_df_user = test_df[test_df["user_id"] == user_id]

        # randomly sample 1 test trajectory per user
        test_trajectory = test_df_user.sample(1, random_state=42).iloc[0]
        # convert the input and target trajectories to context stays and target stay
        context_stays = convert_to_tuple_records(test_trajectory["inputs"])
        target_stay = convert_to_tuple_records(test_trajectory["targets"])[0]

        # select N most recent historical stays
        train_df_user = train_df_user.sort_values(by="trail_id", ascending=True)  # sort by trail_id
        train_df_user = train_df_user.iloc[-args.num_historical_stays :]  # select the most recent N stays

        # convert the input and target trajectories to historical records for prompting
        historical_stays = []
        for _, row in train_df_user.iterrows():
            # convert the input and target trajectories to tuple records for prompting
            historical_stays += convert_to_tuple_records(row["inputs"])
            historical_stays += convert_to_tuple_records(row["targets"])

        prompt_template = prompt_generator(args.prompt_type)
        prompt = prompt_template(historical_stays, context_stays, target_stay)

        result = llm.generate(prompt)
        result["prediction"] = [str(x) for x in result["prediction"][:5]]  # limit to 5 predictions
        result["ground_truth"] = target_stay[3]  # attach ground truth

        with open(output_file_path, "w") as f:
            json.dump(result, f, indent=4)

        return result

    results = thread_map(generate_prediction, users, max_workers=args.num_workers)
    metrics = calculate_metrics(results)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
