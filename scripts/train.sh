set -v
set -e

MT5_PRETRAINED_MODEL=google/mt5-small
BART_PRETRAINED_MODEL=fnlp/bart-base-chinese
CPT_PRETRAINED_MODEL=fnlp/cpt-base
DATA_DIR=X
OUTPUT_MODEL_DIR=X
EPOCHS=100
LR=1e-5

export CUDA_VISIBLE_DEVICES=2 
export WANDB_PROJECT=summary-generation
export TOKENIZERS_PARALLELISM=true

python ../src main.py \
    --pretrained_model ${MT5_PRETRAINED_MODEL} \
    --data_path_prefix ${DATA_DIR} \
    --output_model_dir ${OUTPUT_MODEL_DIR} \
    --SFT \
    --num_epochs ${EPOCHS} \
    --lr 1e-5 \