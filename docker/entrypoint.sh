#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"all"}  # Default if not set
REPO_DIR=${REPO_DIR:-"/workspace/YuE-Interface"}

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS"

source /opt/conda/etc/profile.d/conda.sh
conda activate pyenv

# Define models with source (Hugging Face repo) and destination directory
declare -A MODELS
MODELS=(
    ["xcodec_mini_infer"]="m-a-p/xcodec_mini_infer:/workspace/YuE-Interface/inference/xcodec_mini_infer"
    ["YuE-s1-7B-anneal-en-cot"]="m-a-p/YuE-s1-7B-anneal-en-cot:/workspace/models/YuE-s1-7B-anneal-en-cot"
    ["YuE-s1-7B-anneal-en-icl"]="m-a-p/YuE-s1-7B-anneal-en-icl:/workspace/models/YuE-s1-7B-anneal-en-icl"
    ["YuE-s1-7B-anneal-jp-kr-cot"]="m-a-p/YuE-s1-7B-anneal-jp-kr-cot:/workspace/models/YuE-s1-7B-anneal-jp-kr-cot"
    ["YuE-s1-7B-anneal-jp-kr-icl"]="m-a-p/YuE-s1-7B-anneal-jp-kr-icl:/workspace/models/YuE-s1-7B-anneal-jp-kr-icl"
    ["YuE-s1-7B-anneal-zh-cot"]="m-a-p/YuE-s1-7B-anneal-zh-cot:/workspace/models/YuE-s1-7B-anneal-zh-cot"
    ["YuE-s1-7B-anneal-zh-icl"]="m-a-p/YuE-s1-7B-anneal-zh-icl:/workspace/models/YuE-s1-7B-anneal-zh-icl"
    ["YuE-s2-1B-general"]="m-a-p/YuE-s2-1B-general:/workspace/models/YuE-s2-1B-general"
    ["YuE-upsampler"]="m-a-p/YuE-upsampler:/workspace/models/YuE-upsampler"
)

if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    echo "Installing CUDA nvcc..."
    conda install -y -c nvidia cuda-nvcc --override-channels

    echo "Installing dependencies from requirements.txt..."
    pip install --no-cache-dir -r $REPO_DIR/requirements.txt

    if [ "$DOWNLOAD_MODELS" != "false" ]; then
        echo "Downloading selected models..."

        if [ "$DOWNLOAD_MODELS" = "all" ]; then
            SELECTED_MODELS=("${!MODELS[@]}")
        else
            IFS=',' read -r -a SELECTED_MODELS <<< "$DOWNLOAD_MODELS"
        fi

        for MODEL in "${SELECTED_MODELS[@]}"; do
            if [[ -v MODELS[$MODEL] ]]; then
                SOURCE_DEST=(${MODELS[$MODEL]//:/ })
                SOURCE="${SOURCE_DEST[0]}"
                DESTINATION="${SOURCE_DEST[1]}"

                if [ ! -d "$DESTINATION" ]; then
                    echo "Downloading model: $MODEL from $SOURCE to $DESTINATION"
                    huggingface-cli download "$SOURCE" --local-dir "$DESTINATION"
                else
                    echo "Skipping the model $MODEL because it already exists in $DESTINATION."
                fi
            else
                echo "Warning: Model $MODEL is not recognized. Skipping."
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
export PYTHONPATH="$REPO_DIR:$REPO_DIR/inference/:$PYTHONPATH"
export PATH="$REPO_DIR:$PATH"

echo $PATH
echo $PYTHONPATH

cd /workspace/YuE-Interface

# Use conda python instead of system python
echo "Starting Gradio interface..."
python interface.py &

# Use debugpy for debugging
# exec python -m debugpy --wait-for-client --listen 0.0.0.0:5678 gradio_interface.py

# echo "Starting Tensorboard interface..."
# $CONDA_DIR/bin/conda run -n pyenv tensorboard --logdir_spec=/workspace/outputs --bind_all --port 6006 &
wait