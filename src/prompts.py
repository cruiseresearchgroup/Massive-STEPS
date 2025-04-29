import random

from utils import haversine_distance


def prompt_generator(prompt_type):
    if prompt_type == "llmzs":
        return prompt_generator_llmzs
    elif prompt_type == "llmmob":
        return prompt_generator_llmmob
    elif prompt_type == "llmmove":
        return prompt_generator_llmmove
    elif prompt_type == "llmfs":
        return prompt_generator_llmfs
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


# Modified from https://github.com/tsinghua-fib-lab/AgentMove/blob/main/models/prompts.py
def prompt_generator_llmzs(historical_stays, context_stays, target_stay, **kwargs):
    prompt = f"""Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

Predict <next_place_id> by considering:
1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

Present your answer in a JSON object with:
"prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).

The data:
<historical_stays>: {[[item[0],item[1],item[3]] for item in historical_stays]}
<context_stays>: {[[item[0],item[1],item[3]] for item in context_stays]}
<target_stay>: {[target_stay[0], target_stay[1]]}
"""
    return prompt


def prompt_generator_llmmob(historical_stays, context_stays, target_stay, **kwargs):
    prompt = f"""Your task is to predict a user's next location based on his/her activity pattern.
You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
start_time: the start time of the stay in 12h clock format.
day_of_week: indicating the day of the week.
duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
place_id: an integer representing the unique place ID, which indicates where the stay is.

Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
unknown duration denoted as None, while temporal information is provided.      

Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
1. the activity pattern of this user that you learned from <history>, e.g., repeated visits to certain places during certain times;
2. the context stays in <context>, which provide more recent activities of this user; 
3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
and on different days (e.g., weekday versus weekend).

Please organize your answer in a JSON object containing following keys:
"prediction" (the ID of the five most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

The data are as follows:
<history>: {[[item[0],item[1],item[3]] for item in historical_stays]}
<context>: {[[item[0],item[1],item[3]] for item in context_stays]}
<target_stay>: {[target_stay[0], target_stay[1]]}
    """
    return prompt


# Modified from https://github.com/LLMMove/LLMMove/blob/main/models/LLMMove.py#L36
def prompt_generator_llmmove(historical_stays, context_stays, target_stay, poi_infos, negative_sample_size):
    # get most recent context stay
    most_recent_poi_id = context_stays[-1][3]
    negative_samples = random.sample(sorted(poi_infos.keys()), negative_sample_size)
    candidate_set = negative_samples + [target_stay[3]]
    candidates = [
        (
            candidate_poi_id,
            haversine_distance(
                poi_infos[candidate_poi_id]["latitude"],
                poi_infos[candidate_poi_id]["longitude"],
                poi_infos[most_recent_poi_id]["latitude"],
                poi_infos[most_recent_poi_id]["longitude"],
            ),
            poi_infos[candidate_poi_id]["category"],
        )
        for candidate_poi_id in candidate_set
    ]
    candidates.sort(key=lambda x: x[1])  # sort by distance

    prompt = f"""\
<long-term check-ins> [Format: (POIID, Category)]: {[[item[3], item[2]] for item in historical_stays]}
<recent check-ins> [Format: (POIID, Category)]: {[[item[3], item[2]] for item in context_stays]}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}
Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.

Requirements:
1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
2. Consider the recent check-ins to extract users' current perferences.
3. Consider the "Distance" since people tend to visit nearby pois.
4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.

Please organize your answer in a JSON object containing following keys:
"prediction" (10 distinct POIIDs of the ten most probable places in <candidate set> in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
"""
    return prompt


def prompt_generator_llmfs(
    historical_stays, context_stays, target_stay, poi_infos, negative_sample_size, similar_stays
):
    # get most recent context stay
    most_recent_poi_id = context_stays[-1][3]
    negative_samples = random.sample(sorted(poi_infos.keys()), negative_sample_size)
    candidate_set = negative_samples + [target_stay[3]]
    candidates = [
        (
            candidate_poi_id,
            haversine_distance(
                poi_infos[candidate_poi_id]["latitude"],
                poi_infos[candidate_poi_id]["longitude"],
                poi_infos[most_recent_poi_id]["latitude"],
                poi_infos[most_recent_poi_id]["longitude"],
            ),
            poi_infos[candidate_poi_id]["category"],
        )
        for candidate_poi_id in candidate_set
    ]
    candidates.sort(key=lambda x: x[1])  # sort by distance

    prompt = f"""\
<long-term check-ins> [Format: (POIID, Category)]: {[[item[3], item[2]] for item in historical_stays]}
<similar historical check-ins> [Format: (POIID, Category)]: {[[item[3], item[2]] for item in similar_stays]}
<recent check-ins> [Format: (POIID, Category)]: {[[item[3], item[2]] for item in context_stays]}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}
Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
You have also been provided with <similar historical check-ins>. These check-ins may be from the same user or other users in the past. They might reveal some common travel patterns.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.

Requirements:
1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
2. Consider the recent check-ins to extract users' current perferences.
3. Consider the "Distance" since people tend to visit nearby pois.
4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.

Please organize your answer in a JSON object containing following keys:
"prediction" (10 distinct POIIDs of the ten most probable places in <candidate set> in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
"""
    return prompt
