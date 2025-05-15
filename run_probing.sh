DATA_PATH=data/
DATASET_SAVE_PATH=downloads/places_dataset
ACTIVATION_SAVE_PATH=downloads/activation_datasets

MODEL=meta-llama/Llama-3.2-1B

ACTIVATION_AGGREGATION=last
PROMPT_NAME=empty

python src/make_prompt_dataset.py \
    --data_path $DATA_PATH \
    --model_checkpoint $MODEL \
    --dataset_save_path $DATASET_SAVE_PATH \
    --remove_outliers

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
    --model_checkpoint $MODEL \
    --dataset_save_path $DATASET_SAVE_PATH \
    --activation_save_path $ACTIVATION_SAVE_PATH \
    --activation_aggregation $ACTIVATION_AGGREGATION \
    --prompt_name $PROMPT_NAME

# python src/plot_graphs.py \
#     --input_dir downloads/results/ \
#     --entity_type $ENTITY_TYPE \
#     --activation_aggregation $ACTIVATION_AGGREGATION \
#     --prompt_name $PROMPT_NAME