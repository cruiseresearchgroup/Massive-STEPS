ENTITY_FILE=downloads/nyc_place.csv
ENTITY_TYPE=nyc_place
DATASET_SAVE_PATH=downloads/nyc_place_dataset
ACTIVATION_SAVE_PATH=downloads/activation_datasets

MODEL=meta-llama/Llama-2-7b-hf

ACTIVATION_AGGREGATION=last
PROMPT_NAME=empty

python src/make_prompt_dataset.py \
    --entity_file $ENTITY_FILE \
    --model_checkpoint $MODEL \
    --dataset_save_path $DATASET_SAVE_PATH

python src/save_activations.py \
    --model_checkpoint $MODEL \
    --dataset_save_path $DATASET_SAVE_PATH \
    --activation_save_path $ACTIVATION_SAVE_PATH \
    --activation_aggregation $ACTIVATION_AGGREGATION \
    --prompt_name $PROMPT_NAME \
    --batch_size 8 \
    --save_precision 8 \
    --device cuda

python src/probe_experiment.py \
    --entity_file $ENTITY_FILE \
    --entity_type $ENTITY_TYPE \
    --activation_save_path $ACTIVATION_SAVE_PATH \
    --activation_aggregation $ACTIVATION_AGGREGATION \
    --prompt_name $PROMPT_NAME \
    --model_checkpoint $MODEL

python src/plot_graphs.py \
    --input_dir downloads/ \
    --entity_type $ENTITY_TYPE \
    --activation_aggregation $ACTIVATION_AGGREGATION \
    --prompt_name $PROMPT_NAME