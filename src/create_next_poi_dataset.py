from argparse import ArgumentParser

import pandas as pd
from tqdm.auto import tqdm
from huggingface_hub import HfApi
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

api = HfApi()
tqdm.pandas()

"""
Usage:

python src/create_next_poi_dataset.py \
    --checkins_file data/sydney/sydney_checkins.csv \
    --dataset_id w11wo/STD-Sydney-POI
"""


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkins_file", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.checkins_file)
    max_venue_id = df["venue_id"].max()

    def generate_prompts(group):
        user_id = str(group["user_id"].values[0])
        trail_id = str(group["trail_id"].values[0])

        checkins = group.iloc[:-1]
        checkins.loc[:, "prompt"] = checkins.apply(
            lambda row: f"At {row['timestamp']}, user {user_id} visited POI id {row['venue_id']} which is a {row['venue_category']}, at {row['venue_city']}, {row['venue_country']}.",
            axis=1,
        )
        current_trajectory_prompt = " ".join(checkins["prompt"].values)

        last_checkin = group.iloc[-1]
        prompt = f"The following data is a trajectory of user {user_id}: {current_trajectory_prompt} Given the data, At {last_checkin['timestamp']}, Which POI id will user {user_id} visit? Note that POI id is an integer in the range from 0 to {max_venue_id}."
        answer = f"At {last_checkin['timestamp']}, user {user_id} will visit POI id {last_checkin['venue_id']}."
        return pd.Series([user_id, trail_id, prompt, answer], index=["user_id", "trail_id", "inputs", "targets"])

    results = df.groupby("trail_id").progress_apply(generate_prompts).reset_index(drop=True)

    train_df, test_df = train_test_split(results, stratify=results["user_id"], test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42)

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})
    dataset.push_to_hub(args.dataset_id, private=args.private)
    print(dataset)

    # upload checkin file
    checkins_file = args.checkins_file.split("/")[-1]
    api.upload_file(
        path_or_fileobj=args.checkins_file, path_in_repo=checkins_file, repo_id=args.dataset_id, repo_type="dataset"
    )


if __name__ == "__main__":
    main()
