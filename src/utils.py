import re
from datetime import datetime


def convert_to_tuple_records(inputs):
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
            poi_category_name = None

        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        time_12h = dt.strftime("%I:%M %p")
        day_of_week = dt.strftime("%A")
        records.append((time_12h, day_of_week, poi_category_name, poi_id))

    return records
