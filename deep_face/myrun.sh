#!/bin/bash

SUBDIR_NAME=biden
python -m src.main \
    --input_dir_path ./images/${SUBDIR_NAME} \
    --output_dir_path ./outputs/${SUBDIR_NAME} \

END_POINT=https://2024-05-21-gpt4o.openai.azure.com/
MODEL_NAME=2024-05-21-gpt4o
#MODEL_NAME="2024-09-20-gpt4o"
SECRET_DIR_PATH=/home/kumada/projects/gpt4_vision_demo/
SECRET_INFO_PATH=${SECRET_DIR_PATH}/secret_infos/secret_info_vision.json
TEMPERATURE=0.7
TOP_P=1.0
MAX_TOKENS=1000
ROOT_DIR_PATH=/home/kumada/projects/cct-seiya-kumada/deep_face

#python -m src.main_with_gpt4o \
#    --end_point ${END_POINT} \
#    --secret_info_path ${SECRET_INFO_PATH} \
#    --model_name ${MODEL_NAME} \
#    --input_dir_path  ${ROOT_DIR_PATH}/images/${SUBDIR_NAME} \
#    --output_dir_path ${ROOT_DIR_PATH}/outputs/${SUBDIR_NAME}/ \
#    --max_tokens ${MAX_TOKENS} \
#    --temperature ${TEMPERATURE} \
#    --top_p ${TOP_P} \
