DATA_PATH=data/
DATASET_SAVE_PATH=downloads/places_dataset
ACTIVATION_SAVE_PATH=downloads/activation_datasets
OUTPUT_DIR=probing_results

MODEL=meta-llama/Llama-3.2-1B

ACTIVATION_AGGREGATION=last
PROMPT_NAME=address_city

for city in beijing istanbul jakarta kuwait_city melbourne moscow new_york petaling_jaya sao_paulo shanghai sydney tokyo; do
    DATASET_SAVE_PATH_CITY=${DATASET_SAVE_PATH}_${city}
    ACTIVATION_SAVE_PATH_CITY=${ACTIVATION_SAVE_PATH}_${city}

    python src/make_prompt_dataset.py \
        --data_path $DATA_PATH \
        --model_checkpoint $MODEL \
        --dataset_save_path $DATASET_SAVE_PATH_CITY \
        --remove_outliers \
        --city $city \
        --prompt_name $PROMPT_NAME

    python src/save_activations.py \
        --model_checkpoint $MODEL \
        --dataset_save_path $DATASET_SAVE_PATH_CITY \
        --activation_save_path $ACTIVATION_SAVE_PATH_CITY \
        --activation_aggregation $ACTIVATION_AGGREGATION \
        --prompt_name $PROMPT_NAME \
        --batch_size 8 \
        --save_precision 8 \
        --device cuda

    python src/probe_experiment.py \
        --model_checkpoint $MODEL \
        --dataset_save_path $DATASET_SAVE_PATH_CITY \
        --activation_save_path $ACTIVATION_SAVE_PATH_CITY \
        --activation_aggregation $ACTIVATION_AGGREGATION \
        --prompt_name $PROMPT_NAME \
        --city $city \
        --output_dir $OUTPUT_DIR
done

python src/plot_graphs.py \
    --input_dir $OUTPUT_DIR \
    --activation_aggregation $ACTIVATION_AGGREGATION \
    --prompt_name $PROMPT_NAME \
    --model_checkpoint $MODEL