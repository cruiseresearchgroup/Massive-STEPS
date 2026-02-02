from argparse import ArgumentParser
from typing import Literal
from pathlib import Path
import json

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np

from utils import convert_to_tuple_records, get_poi_infos


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--checkins_file", type=str, required=True)
    # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/run_fsq.sh#L1
    parser.add_argument("--num_users", type=int, default=200)
    parser.add_argument("--vectorizer", type=str, choices=["tfidf", "bow"])
    parser.add_argument("--model_name", type=str, choices=["logistic_regression", "xgboost", "random_forest"])
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = args.model_name.split("/")[-1]
    output_dir = args.output_dir / dataset_name / "classical" / "st_day_classification" / model_name / args.vectorizer
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(token_pattern=r"\S+")
    elif args.vectorizer == "bow":
        vectorizer = CountVectorizer(token_pattern=r"\S+")

    if args.model_name == "logistic_regression":
        model = LogisticRegression(max_iter=200)
    elif args.model_name == "xgboost":
        model = XGBClassifier(n_estimators=100, max_depth=3)
    elif args.model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=3)

    dataset = load_dataset(args.dataset_name)
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    users = sorted(test_df["user_id"].unique())
    users = np.random.RandomState(args.seed).choice(users, min(args.num_users, len(users)), replace=False)
    poi_infos = get_poi_infos(args.checkins_file)

    # randomly sample 1 trajectory per user for both train and test sets
    train_df = train_df[train_df["user_id"].isin(users)]
    test_df = test_df[test_df["user_id"].isin(users)]
    train_df = train_df.groupby("user_id").sample(n=1, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.groupby("user_id").sample(n=1, random_state=args.seed).reset_index(drop=True)

    def convert_trajectory_to_category_sequences(inputs: str, targets: str) -> tuple[str, int]:
        # convert the input and target trajectories to context stays and target stay
        context_stays = convert_to_tuple_records(inputs, poi_infos)
        target_stay = convert_to_tuple_records(targets, poi_infos)[0]
        trajectory = context_stays + [target_stay]

        # category is the 3rd element in the tuple
        category_sequence = [stay[2] for stay in trajectory]
        # join multi-word categories with underscore
        # then join the sequence with space
        category_sequence = " ".join(["_".join(category.split()) for category in category_sequence])

        # label is 1 if the target stay is on weekend, else 0
        label = 1 if target_stay[1] in ("Saturday", "Sunday") else 0
        return category_sequence, label

    train_category_sequences = train_df.apply(
        lambda row: convert_trajectory_to_category_sequences(row["inputs"], row["targets"]), axis=1
    ).tolist()
    test_category_sequences = test_df.apply(
        lambda row: convert_trajectory_to_category_sequences(row["inputs"], row["targets"]), axis=1
    ).tolist()

    X_train, y_train = zip(*train_category_sequences)
    X_test, y_test = zip(*test_category_sequences)

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model.fit(X_train_vectorized, y_train)
    metrics = {"acc_1": model.score(X_test_vectorized, y_test)}

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
