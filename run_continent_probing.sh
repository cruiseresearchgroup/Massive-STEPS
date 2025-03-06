ENTITY_FILE=downloads/world_place.csv
ENTITY_TYPE=world_place
DATASET_SAVE_PATH=downloads/${ENTITY_TYPE}_dataset
ACTIVATION_SAVE_PATH=downloads/activation_datasets

MODEL=meta-llama/Llama-3.2-1B

ACTIVATION_AGGREGATION=last
PROMPT_NAME=empty

for CONTINENT in $(cat downloads/continents.txt)
do
    DATASET_SAVE_PATH_CONT=${DATASET_SAVE_PATH}_${CONTINENT}
    ENTITY_TYPE_CONT=${ENTITY_TYPE}_${CONTINENT}

    python src/make_prompt_dataset.py \
        --entity_file $ENTITY_FILE \
        --entity_type $ENTITY_TYPE_CONT \
        --model_checkpoint $MODEL \
        --dataset_save_path $DATASET_SAVE_PATH_CONT \
        --continent $CONTINENT \
        --remove_outliers

    python src/save_activations.py \
        --model_checkpoint $MODEL \
        --dataset_save_path $DATASET_SAVE_PATH_CONT \
        --activation_save_path $ACTIVATION_SAVE_PATH \
        --activation_aggregation $ACTIVATION_AGGREGATION \
        --entity_type $ENTITY_TYPE_CONT \
        --prompt_name $PROMPT_NAME \
        --batch_size 8 \
        --save_precision 8 \
        --device cuda

    python src/probe_experiment.py \
        --entity_file $ENTITY_FILE \
        --entity_type $ENTITY_TYPE_CONT \
        --activation_save_path $ACTIVATION_SAVE_PATH \
        --activation_aggregation $ACTIVATION_AGGREGATION \
        --prompt_name $PROMPT_NAME \
        --model_checkpoint $MODEL \
        --continent $CONTINENT \
        --remove_outliers

    python src/plot_graphs.py \
        --input_dir downloads/results/ \
        --entity_type $ENTITY_TYPE_CONT \
        --activation_aggregation $ACTIVATION_AGGREGATION \
        --prompt_name $PROMPT_NAME
done