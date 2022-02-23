set -v
set -e

MODEL_PATH=
MODEL_NAME=cpt
export CUDA_VISIBLE_DEVICES=2 

python ../src predict.py \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME}