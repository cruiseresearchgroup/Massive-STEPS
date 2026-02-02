for city in Bandung Beijing Istanbul Jakarta Kuwait-City Melbourne Moscow New-York Palembang Petaling-Jaya Sao-Paulo Shanghai Sydney Tangerang Tokyo; do
    city_key=$(echo "$city" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    for model in logistic_regression random_forest xgboost; do
        for vectorizer in tfidf bow; do
            echo "Running classification for $city with $model and $vectorizer"
            python src/run_classify_day_classical.py \
                --dataset_name CRUISEResearchGroup/Massive-STEPS-$city \
                --num_users 200  \
                --vectorizer $vectorizer \
                --model_name $model \
                --city $city \
                --checkins_file data/$city_key/${city_key}_checkins.csv
        done
    done
done