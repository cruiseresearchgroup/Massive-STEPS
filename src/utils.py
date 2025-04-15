import re
import csv
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return round(distance, 3)


def convert_to_tuple_records(inputs, poi_infos=None):
    # Extracting the timestamp and POI id from the input string
    # following the prompt format of AgentMove:
    # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/processing/data.py#L259
    # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/models/prompts.py

    is_input_trajectory = inputs.startswith("The following data is a trajectory")
    records = []

    if is_input_trajectory:  # input trajectory
        pattern = (
            r"At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user \d+ visited POI id (\d+) which is a (.*?), at .*?, .*?\."
        )
    else:  # target trajectory
        pattern = r"At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user \d+ will visit POI id (\d+)\."

    matches = re.findall(pattern, inputs)

    for match in matches:
        if len(match) == 3:  # input trajectory
            timestamp, poi_id, poi_category_name = match
        else:  # target trajectory
            timestamp, poi_id = match
            poi_category_name = poi_infos[poi_id]["category"]

        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        time_12h = dt.strftime("%I:%M %p")
        day_of_week = dt.strftime("%A")
        records.append((time_12h, day_of_week, poi_category_name, poi_id))

    return records


def get_poi_infos(checkins_file):
    poi_infos = {}
    with open(checkins_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poi_id = str(row["venue_id"])
            if poi_id not in poi_infos:
                poi_infos[poi_id] = {
                    "latitude": float(row["venue_city_latitude"]),
                    "longitude": float(row["venue_city_longitude"]),
                    "category": row["venue_category"],
                }
    return poi_infos
