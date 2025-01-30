#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"all_bf16"}  # Default if not set, all, all_int8, all_bf16, or comma-separated list of models
REPO_DIR=${REPO_DIR:-"/workspace/YuE-Interface"}
MODEL_DIR=${MODEL_DIR:-"/workspace/models"}

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS"

source /opt/conda/etc/profile.d/conda.sh
conda activate pyenv

# Define models with source (Hugging Face repo) and destination directory
declare -A MODELS_BF16
MODELS_BF16["YuE-s1-7B-anneal-en-cot"]="m-a-p/YuE-s1-7B-anneal-en-cot:${MODEL_DIR}/YuE-s1-7B-anneal-en-cot"
MODELS_BF16["YuE-s1-7B-anneal-en-icl"]="m-a-p/YuE-s1-7B-anneal-en-icl:${MODEL_DIR}/YuE-s1-7B-anneal-en-icl"
MODELS_BF16["YuE-s1-7B-anneal-jp-kr-cot"]="m-a-p/YuE-s1-7B-anneal-jp-kr-cot:${MODEL_DIR}/YuE-s1-7B-anneal-jp-kr-cot"
MODELS_BF16["YuE-s1-7B-anneal-jp-kr-icl"]="m-a-p/YuE-s1-7B-anneal-jp-kr-icl:${MODEL_DIR}/YuE-s1-7B-anneal-jp-kr-icl"
MODELS_BF16["YuE-s1-7B-anneal-zh-cot"]="m-a-p/YuE-s1-7B-anneal-zh-cot:${MODEL_DIR}/YuE-s1-7B-anneal-zh-cot"
MODELS_BF16["YuE-s1-7B-anneal-zh-icl"]="m-a-p/YuE-s1-7B-anneal-zh-icl:${MODEL_DIR}/YuE-s1-7B-anneal-zh-icl"
MODELS_BF16["YuE-s2-1B-general"]="m-a-p/YuE-s2-1B-general:${MODEL_DIR}/YuE-s2-1B-general"
MODELS_BF16["YuE-upsampler"]="m-a-p/YuE-upsampler:${MODEL_DIR}/YuE-upsampler"

declare -A MODELS_INT8
MODELS_INT8["YuE-s1-7B-anneal-en-cot-int8"]="Alissonerdx/YuE-s1-7B-anneal-en-cot-int8:${MODEL_DIR}/YuE-s1-7B-anneal-en-cot-int8"
MODELS_INT8["YuE-s1-7B-anneal-en-icl-int8"]="Alissonerdx/YuE-s1-7B-anneal-en-icl-int8:${MODEL_DIR}/YuE-s1-7B-anneal-en-icl-int8"
MODELS_INT8["YuE-s1-7B-anneal-jp-kr-cot-int8"]="Alissonerdx/YuE-s1-7B-anneal-jp-kr-cot-int8:${MODEL_DIR}/YuE-s1-7B-anneal-jp-kr-cot-int8"
MODELS_INT8["YuE-s1-7B-anneal-jp-kr-icl-int8"]="Alissonerdx/YuE-s1-7B-anneal-jp-kr-icl-int8:${MODEL_DIR}/YuE-s1-7B-anneal-jp-kr-icl-int8"
MODELS_INT8["YuE-s1-7B-anneal-zh-cot-int8"]="Alissonerdx/YuE-s1-7B-anneal-zh-cot-int8:${MODEL_DIR}/YuE-s1-7B-anneal-zh-cot-int8"
MODELS_INT8["YuE-s1-7B-anneal-zh-icl-int8"]="Alissonerdx/YuE-s1-7B-anneal-zh-icl-int8:${MODEL_DIR}/YuE-s1-7B-anneal-zh-icl-int8"
MODELS_INT8["YuE-s2-1B-general-int8"]="Alissonerdx/YuE-s2-1B-general-int8:${MODEL_DIR}/YuE-s2-1B-general-int8"
MODELS_INT8["YuE-upsampler"]="m-a-p/YuE-upsampler:${MODEL_DIR}/YuE-upsampler"



if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    echo "Installing CUDA nvcc..."
    conda install -y -c nvidia cuda-nvcc --override-channels

    echo "Installing dependencies from requirements.txt..."
    pip install --no-cache-dir -r $REPO_DIR/requirements.txt

    if [ ! -d "${REPO_DIR}/inference/xcodec_mini_infer" ]; then
        huggingface-cli download m-a-p/xcodec_mini_infer --local-dir "${REPO_DIR}/inference/xcodec_mini_infer"
    else
        echo "Skipping the model xcodec_mini_infer download because it already exists."
    fi

    if [ "$DOWNLOAD_MODELS" != "false" ]; then
        echo "Downloading selected models..."

        if [ "$DOWNLOAD_MODELS" = "all_bf16" ]; then
            SELECTED_MODELS=("${!MODELS_BF16[@]}")
            declare -A MODELS
            for key in "${!MODELS_BF16[@]}"; do
                MODELS[$key]="${MODELS_BF16[$key]}"
            done
        elif [ "$DOWNLOAD_MODELS" = "all_int8" ]; then
            SELECTED_MODELS=("${!MODELS_INT8[@]}")
            declare -A MODELS
            for key in "${!MODELS_INT8[@]}"; do
                MODELS[$key]="${MODELS_INT8[$key]}"
            done
        elif [ "$DOWNLOAD_MODELS" = "all" ]; then
            SELECTED_MODELS=("${!MODELS_BF16[@]}" "${!MODELS_INT8[@]}")
            declare -A MODELS
            for key in "${!MODELS_BF16[@]}"; do
                MODELS[$key]="${MODELS_BF16[$key]}"
            done
            for key in "${!MODELS_INT8[@]}"; do
                MODELS[$key]="${MODELS_INT8[$key]}"
            done
        else
            IFS=',' read -r -a SELECTED_MODELS <<< "$DOWNLOAD_MODELS"
            declare -A MODELS
            for MODEL in "${SELECTED_MODELS[@]}"; do
                if [[ -v MODELS_BF16[$MODEL] ]]; then
                    MODELS[$MODEL]="${MODELS_BF16[$MODEL]}"
                elif [[ -v MODELS_INT8[$MODEL] ]]; then
                    MODELS[$MODEL]="${MODELS_INT8[$MODEL]}"
                else
                    echo "Warning: Model $MODEL is not recognized. Skipping."
                fi
            done
        fi
        for MODEL in "${SELECTED_MODELS[@]}"; do
            SOURCE_DEST=(${MODELS[$MODEL]//:/ })
            SOURCE="${SOURCE_DEST[0]}"
            DESTINATION="${SOURCE_DEST[1]}"

            if [ ! -d "$DESTINATION" ]; then
                echo "Downloading model: $MODEL from $SOURCE to $DESTINATION"
                huggingface-cli download "$SOURCE" --local-dir "$DESTINATION"
            else
                echo "Skipping the model $MODEL because it already exists in $DESTINATION."
            fi
        done
    else
        echo "DOWNLOAD_MODELS is false, skipping model downloads."
    fi

    # Create marker file
    touch "$INIT_MARKER"
    echo "Initialization complete."
else
    echo "Container already initialized. Skipping first-time setup."
fi

echo "Adding environment variables"
export PATH="$REPO_DIR:$REPO_DIR/inference/:$REPO_DIR/inference/xcodec_mini_infer:$REPO_DIR/inference/xcodec_mini_infer:$REPO_DIR/inference/xcodec_mini_infer/descriptaudiocodec:$PATH"

echo $PATH
echo $PYTHONPATH

cd /workspace/YuE-Interface/inference

# Use conda python instead of system python
echo "Starting Gradio interface..."
python interface.py &

# Use debugpy for debugging
# exec python -m debugpy --wait-for-client --listen 0.0.0.0:5678 gradio_interface.py

# echo "Starting Tensorboard interface..."
# $CONDA_DIR/bin/conda run -n pyenv tensorboard --logdir_spec=/workspace/outputs --bind_all --port 6006 &
wait
