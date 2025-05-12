import re
import csv
import json
from pathlib import Path
from dateutil import parser
from argparse import ArgumentParser

import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from shapely.geometry import Point, shape

STD_2013_LINES = 18587049
STD_2018_LINES = 11910007

"""
Usage:

python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/sydney/sydney_5750005_overpass.geojson \
    --output_dir data/sydney \
    --output_file sydney_checkins.csv \
    --min_checkins 2 --min_trails 3
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--std_2013_file", type=str, default="semantic-trails/downloads/std_2013.csv")
    parser.add_argument("--std_2018_file", type=str, default="semantic-trails/downloads/std_2018.csv")
    parser.add_argument("--cities_file", type=str, default="semantic-trails/cities.csv")
    parser.add_argument("--categories_file", type=str, default="semantic-trails/categories.csv")
    parser.add_argument("--city_geo_json_file", type=str, required=True)
    parser.add_argument("--min_trails", type=int, default=2)
    parser.add_argument("--min_checkins", type=int, default=2)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_file

    fsq_ds = load_dataset("foursquare/fsq-os-places", "places", split="train")
    keep_columns = ["fsq_place_id", "name", "latitude", "longitude", "address"]
    fsq_ds = fsq_ds.remove_columns(list(set(fsq_ds.column_names) - set(keep_columns)))

    # map category id to name
    category2name = {}

    with open(args.categories_file, "r") as f:
        lines = csv.reader(f, delimiter=",")
        for line in lines:
            category, name, *_ = line
            category2name[category] = name

    # load city GeoJSON to filter administrative area based on lat/lon
    with open(args.city_geo_json_file) as f:
        city_geo = json.load(f)
    city_polygon = shape(city_geo["features"][0]["geometry"])

    cities = set()
    city2detail = {}

    with open(args.cities_file, "r") as f:
        lines = csv.reader(f, delimiter=",")
        headers = next(lines)
        for line in lines:
            lat, lon, _, _, _, cc = line
            point = Point(float(lon), float(lat))
            if city_polygon.intersects(point):
                cities.add(cc)
                city2detail[cc] = dict(zip(headers, line))

    trajectories = []
    venues, categories = set(), set()

    for std_file in [args.std_2013_file, args.std_2018_file]:
        with open(std_file, "r") as f:
            lines = csv.reader(f, delimiter=",")
            headers = next(lines)
            total = STD_2013_LINES if "2013" in std_file else STD_2018_LINES
            subset = "2013" if "2013" in std_file else "2018"
            for line in tqdm(lines, total=total):
                trail_id, _, venue_id, venue_category, _, city, _, _ = line
                line[0] = f"{subset}_{trail_id}"  # add subset prefix to trail_id
                venue_id = venue_id.replace("foursquare:", "")  # remove foursquare: prefix
                line[2] = venue_id
                city = city.replace("wd:", "")  # remove wd: prefix
                # filter checkins based on city
                if city not in cities:
                    continue

                trajectories.append(line)
                venues.add(venue_id)
                categories.add(venue_category)

    fsq_df = fsq_ds.filter(lambda x: x in venues, input_columns=["fsq_place_id"], num_proc=16).to_pandas()

    venue2index = {venue: i for i, venue in enumerate(sorted(venues))}
    category2index = {category: i for i, category in enumerate(sorted(categories))}

    df = pd.DataFrame(trajectories, columns=headers)
    df = df.merge(fsq_df, how="left", left_on="venue_id", right_on="fsq_place_id")  # merge with FourSquare dataset
    df["timestamp"] = df["timestamp"].apply(lambda t: str(parser.isoparse(t).replace(tzinfo=None)))  # remove offset
    df["venue_category_id"] = df["venue_category"]  # keep original category code
    df["venue_category_id_code"] = df["venue_category"].map(category2index)  # map/anonymize category to index
    df["venue_category"] = df["venue_category"].map(category2name)  # get POI category name
    df["venue_id"] = df["venue_id"].map(dict(venue2index))  # anonymize POI ids
    df["venue_city"] = df["venue_city"].apply(lambda x: x.replace("wd:", ""))  # remove wd: prefix
    df["venue_country"] = df["venue_city"].map(lambda city: city2detail[city]["admin2"])  # get country name
    df["venue_city_latitude"] = df["venue_city"].map(lambda city: city2detail[city]["lat"])  # get city latitude
    df["venue_city_longitude"] = df["venue_city"].map(lambda city: city2detail[city]["lon"])  # get city longitude
    df["venue_city"] = df["venue_city"].map(lambda city: city2detail[city]["admin1"])  # get city name
    df["name"] = df["name"].apply(lambda x: re.sub(r"[\n\r]", " ", x) if isinstance(x, str) else x)
    df["address"] = df["address"].apply(lambda x: re.sub(r"[\n\r]", " ", x) if isinstance(x, str) else x)

    columns = [
        "trail_id",
        "user_id",
        "venue_id",
        "latitude",
        "longitude",
        "name",
        "address",
        "venue_category",
        "venue_category_id",
        "venue_category_id_code",
        "venue_city",
        "venue_city_latitude",
        "venue_city_longitude",
        "venue_country",
        "timestamp",
    ]
    df = df[columns]

    # filter trails with < 2 checkins
    trail_checkins = df.groupby("trail_id").size()
    trail_checkins = trail_checkins[trail_checkins >= args.min_checkins].index
    df = df[df["trail_id"].isin(trail_checkins)]

    # filter users with < 2 trails
    user_trails = df.groupby("user_id")["trail_id"].nunique()
    user_trails = user_trails[user_trails >= args.min_trails].index
    df = df[df["user_id"].isin(user_trails)]

    df.to_csv(output_file, index=False)

    # save checkin statistics
    stats = {
        "num_checkins": len(df),
        "num_users": df["user_id"].nunique(),
        "num_venues": len(venues),
        "num_trails": df["trail_id"].nunique(),
    }

    with open(output_file.with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main(parse_args())
