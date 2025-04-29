from argparse import ArgumentParser
from pathlib import Path
import json

from datasets import load_dataset
from tqdm.contrib.concurrent import thread_map
from sentence_transformers import SentenceTransformer
import numpy as np

from prompts import prompt_generator
from llm import Gemini, vLLM
from utils import convert_to_tuple_records, get_poi_infos, extract_timestamp, create_historical_trajectory_prompt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--checkins_file", type=str, required=True)
    # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/run_fsq.sh#L1
    parser.add_argument("--num_users", type=int, default=200)
    # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/utils.py#L168
    parser.add_argument("--num_historical_stays", type=int, default=15)
    # https://github.com/LLMMove/LLMMove/blob/main/models/LLMMove.py#L49
    parser.add_argument("--negative_sample_size", type=int, default=100)
    # LLM-FS parameters
    parser.add_argument("--sentence_model_checkpoint", type=str, default="nomic-ai/modernbert-embed-base")
    parser.add_argument("--top_k_similar_trajectories", type=int, default=5)
    parser.add_argument("--encode_batch_size", type=int, default=1)
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
        ndcg_5 += (1 / np.log2(predictions.index(ground_truth) + 1 + 1)) if ground_truth in predictions else 0

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
    poi_infos = get_poi_infos(args.checkins_file)

    if "gemini" in args.model_name:
        llm = Gemini(model=args.model_name)
    else:
        llm = vLLM(model=args.model_name)

    if args.prompt_type == "llmfs":
        train_df["timestamp"] = train_df["targets"].apply(extract_timestamp)

        train_df["prompt"] = train_df.apply(create_historical_trajectory_prompt, axis=1, is_train=True)
        test_df["prompt"] = test_df.apply(create_historical_trajectory_prompt, axis=1, is_train=False)

        model = SentenceTransformer(args.sentence_model_checkpoint)
        # embed train prompts as document embeddings
        train_trajectory_embeddings = model.encode(
            [f"search_document: {prompt}" for prompt in train_df["prompt"]],
            show_progress_bar=True,
            batch_size=args.encode_batch_size,
        )
        # embed test prompts as query embeddings
        test_trajectory_embeddings = model.encode(
            [f"search_query: {prompt}" for prompt in test_df["prompt"]],
            show_progress_bar=True,
            batch_size=args.encode_batch_size,
        )

    def generate_prediction(user_id):
        output_file_path = output_dir / f"user_{user_id}.json"

        if output_file_path.exists():
            with open(output_file_path, "r") as f:
                result = json.load(f)
            return result

        train_df_user = train_df[train_df["user_id"] == user_id]
        test_df_user = test_df[test_df["user_id"] == user_id]

        # randomly sample 1 test trajectory per user
        test_trajectory = test_df_user.sample(1, random_state=42).iloc[0]
        # convert the input and target trajectories to context stays and target stay
        context_stays = convert_to_tuple_records(test_trajectory["inputs"], poi_infos)
        target_stay = convert_to_tuple_records(test_trajectory["targets"], poi_infos)[0]

        # select N most recent historical stays
        train_df_user = train_df_user.sort_values(by="trail_id", ascending=True)  # sort by trail_id
        train_df_user = train_df_user.iloc[-args.num_historical_stays :]  # select the most recent N stays

        # convert the input and target trajectories to historical records for prompting
        historical_stays = []
        for _, row in train_df_user.iterrows():
            # convert the input and target trajectories to tuple records for prompting
            historical_stays += convert_to_tuple_records(row["inputs"], poi_infos)
            historical_stays += convert_to_tuple_records(row["targets"], poi_infos)

        prompt_template = prompt_generator(args.prompt_type)
        if args.prompt_type == "llmmove":
            kwargs = {"poi_infos": poi_infos, "negative_sample_size": args.negative_sample_size}
        elif args.prompt_type == "llmfs":
            # embed test trajectory prompt and calculate similarity with train trajectory embeddings
            test_trajectory_embedding = test_trajectory_embeddings[test_trajectory.name, :]
            similarities = model.similarity(test_trajectory_embedding, train_trajectory_embeddings).numpy().squeeze(0)

            # get all trajectories that occur before the test trajectory (avoid data leakage)
            current_timestamp = extract_timestamp(test_trajectory["targets"])
            historical_trajectories_indices = train_df[train_df["timestamp"] < current_timestamp].index

            # get the top K most similar and strictly historical trajectories
            sorted_indices = np.argsort(similarities)[::-1]
            historical_indices = [i for i in sorted_indices if i in historical_trajectories_indices]
            top_k_indices = historical_indices[: args.top_k_similar_trajectories]
            similar_trajectories = train_df.iloc[top_k_indices]
            similar_stays = []
            for _, row in similar_trajectories.iterrows():
                # convert the input and target trajectories to tuple records for prompting
                similar_stays += convert_to_tuple_records(row["inputs"], poi_infos)
                similar_stays += convert_to_tuple_records(row["targets"], poi_infos)

            kwargs = {
                "poi_infos": poi_infos,
                "negative_sample_size": args.negative_sample_size,
                "similar_stays": similar_stays,
            }
        else:
            kwargs = {}

        prompt = prompt_template(historical_stays, context_stays, target_stay, **kwargs)

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
