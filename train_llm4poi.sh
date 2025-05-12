model="unsloth/Llama-3.2-1B-bnb-4bit"

for city in Beijing Istanbul Jakarta Kuwait-City Melbourne Moscow New-York Petaling-Jaya Sao-Paulo Shanghai Sydney Tokyo; do
    $PYTHON_EXEC src/train_llm4poi.py \
        --model_checkpoint $model \
        --max_length 16384 \
        --batch_size 8 \
        --learning_rate 2e-4 \
        --num_epochs 3 \
        --dataset_id w11wo/STD-$city-POI
done