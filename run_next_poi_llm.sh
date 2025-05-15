model=gemini-2.0-flash

for prompt_type in llmzs llmmob llmmove; do
    for city in Beijing Istanbul Jakarta Kuwait-City Melbourne Moscow New-York Petaling-Jaya Sao-Paulo Shanghai Sydney Tokyo; do
        city_key=$(echo "$city" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

        python src/run_next_poi_llm.py \
            --dataset_name w11wo/Massive-STEPS-$city \
            --num_users 200 --num_historical_stays 15 \
            --prompt_type $prompt_type \
            --model_name $model \
            --checkins_file data/$city_key/${city_key}_checkins.csv
    done
done