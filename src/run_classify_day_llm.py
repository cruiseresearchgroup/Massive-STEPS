from argparse import ArgumentParser
from typing import Literal
from pathlib import Path
import json

from pydantic import BaseModel
from datasets import load_dataset
from tqdm.contrib.concurrent import thread_map
import numpy as np

from prompts import prompt_generator
from llm import LLM, Gemini, vLLM
from utils import convert_to_tuple_records, get_poi_infos


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--checkins_file", type=str, required=True)
    # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/run_fsq.sh#L1
    parser.add_argument("--num_users", type=int, default=200)
    parser.add_argument("--prompt_type", type=str, default="st_day_classification")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def calculate_metrics(results):
    acc_1 = 0
    for result in results:
        predictions, ground_truth = result["prediction"], result["ground_truth"]
        acc_1 += 1 if len(predictions) > 0 and predictions[0] == ground_truth else 0

    return {"acc_1": acc_1 / len(results)}


class DayOfWeek(BaseModel):
    prediction: Literal["weekday", "weekend"]
    reason: str


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
    poi_infos = get_poi_infos(args.checkins_file)

    if "gpt" in args.model_name:
        llm = LLM(model=args.model_name)
    elif "gemini" in args.model_name:
        llm = Gemini(model=args.model_name)
    else:
        llm = vLLM(model=args.model_name)

    if "gpt-5" in args.model_name:
        kwargs = {"temperature": 1.0, "reasoning_effort": "medium", "verbosity": "low", "max_completion_tokens": 4096}
        response_format = None
    else:
        kwargs = {}
        response_format = DayOfWeek

    def generate_prediction(user_id):
        output_file_path = output_dir / f"user_{user_id}.json"

        if output_file_path.exists():
            with open(output_file_path, "r") as f:
                result = json.load(f)
            return result

        test_df_user = test_df[test_df["user_id"] == user_id]

        # randomly sample 1 test trajectory per user
        test_trajectory = test_df_user.sample(1, random_state=42).iloc[0]
        # convert the input and target trajectories to context stays and target stay
        context_stays = convert_to_tuple_records(test_trajectory["inputs"], poi_infos)
        target_stay = convert_to_tuple_records(test_trajectory["targets"], poi_infos)[0]

        prompt_template = prompt_generator(args.prompt_type)
        prompt = prompt_template(context_stays, target_stay, city=args.city)

        result = llm.generate(prompt, response_format=response_format, **kwargs)
        result["prediction"] = [result["prediction"]]
        result["ground_truth"] = "weekend" if target_stay[1] in ("Saturday", "Sunday") else "weekday"

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
