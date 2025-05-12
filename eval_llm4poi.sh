model="Llama-3.2-1B-bnb-4bit"

for city in Beijing Istanbul Jakarta Kuwait-City Melbourne Moscow New-York Petaling-Jaya Sao-Paulo Shanghai Sydney Tokyo; do
    $PYTHON_EXEC src/eval_llm4poi.py \
        --model_checkpoint $model-STD-$city-POI \
        --dataset_id w11wo/STD-$city-POI
done