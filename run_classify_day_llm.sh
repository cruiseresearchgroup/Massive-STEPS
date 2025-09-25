model=gemini-2.0-flash

for city in Bandung Beijing Istanbul Jakarta Kuwait-City Melbourne Moscow New-York Palembang Petaling-Jaya Sao-Paulo Shanghai Sydney Tangerang Tokyo; do
    city_key=$(echo "$city" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

    python src/run_classify_day_llm.py \
        --dataset_name YOURUSERNAME/Massive-STEPS-$city \
        --num_users 200  \
        --prompt_type st_day_classification \
        --model_name $model \
        --city $city \
        --checkins_file data/$city_key/${city_key}_checkins.csv
done