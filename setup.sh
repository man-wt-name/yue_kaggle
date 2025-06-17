#!/bin/bash

#-----------------------------------------------------------------------------------------
# setup.sh: ПОЛНЫЙ скрипт установки для проекта Yue-Kaggle в среде Kaggle/Colab.
#
# Версия 2.0: Включает все зависимости из Dockerfile (apt-get, conda, pip)
# и логику скачивания моделей из entrypoint.sh.
#
# Перед запуском:
# Убедитесь, что вы ВРУЧНУЮ создали окружение Conda:
# > conda create -n pyenv python=3.12 -y
#
# Действия скрипта:
# 1. Установка системных зависимостей через apt-get.
# 2. Установка зависимостей Conda в окружение 'pyenv'.
# 3. Установка зависимостей Pip в окружение 'pyenv'.
# 4. Скачивание AI-моделей с Hugging Face.
# 5. Патчинг библиотеки 'transformers' кастомными файлами.
# 6. Создание файла 'env_vars.sh' для установки переменных окружения.
#-----------------------------------------------------------------------------------------

# --- НАСТРОЙКА ---
# Прерываем выполнение скрипта в случае ошибки
set -e
# Название вашего окружения Conda
CONDA_ENV_NAME="pyenv"
# Какие модели скачивать? Варианты:
# "all_bf16", "all_int8", "all_nf4", "all", "false"
# или через запятую, например "YuE-s2-1B-general,YuE-upsampler"
DOWNLOAD_MODELS="false"
# Директория для моделей
MODEL_DIR="/kaggle/working/models"
# Директория репозитория
REPO_DIR="/kaggle/working/yue_kaggle"
HF_TOKEN="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"

# --- ШАГ 1: УСТАНОВКА СИСТЕМНЫХ ЗАВИСИМОСТЕЙ (APT-GET) ---
echo "===== ШАГ 1: Установка системных зависимостей через apt-get... ====="
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates git curl build-essential cmake jq \
    libcurl4-openssl-dev libglib2.0-0 libgl1-mesa-glx libsm6 libssl-dev \
    libxext6 libxrender-dev software-properties-common openssh-server \
    openssh-client git-lfs vim zip unzip zlib1g-dev libc6-dev
# Очистка
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
# Активация git-lfs
git lfs install
echo "===== Системные зависимости установлены. ====="
echo

# --- ШАГ 2: УСТАНОВКА ЗАВИСИМОСТЕЙ CONDA ---
echo "===== ШАГ 2: Установка зависимостей Conda в окружение '$CONDA_ENV_NAME'... ====="
# Проверка существования окружения
if ! conda info --envs | grep -q "^$CONDA_ENV_NAME\s"; then
    echo "ОШИБКА: Окружение Conda '$CONDA_ENV_NAME' не найдено."
    echo "Пожалуйста, создайте его вручную: conda create -n $CONDA_ENV_NAME python=3.12 -y"
    exit 1
fi
conda install -n $CONDA_ENV_NAME -c conda-forge openmpi mpi4py -y
conda install -n $CONDA_ENV_NAME -c nvidia cuda-nvcc -y
echo "===== Зависимости Conda установлены. ====="
echo

# --- ШАГ 3: УСТАНОВКА ЗАВИСИМОСТЕЙ PIP ---
echo "===== ШАГ 3: Установка зависимостей Pip в окружение '$CONDA_ENV_NAME'... ====="
# PyTorch для CUDA 12.4 (версии из Dockerfile)
conda run -n $CONDA_ENV_NAME pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Jupyter и Hugging Face Hub
#conda run -n $CONDA_ENV_NAME pip install jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions nodejs "huggingface_hub[cli]"
# Зависимости проекта
if [ -f "requirements.txt" ]; then
    conda run -n $CONDA_ENV_NAME pip install --no-cache-dir -r requirements.txt
else
    echo "ОШИБКА: Файл requirements.txt не найден!"
    exit 1
fi
echo "===== Зависимости Pip установлены. ====="
echo

# --- ШАГ 4: СКАЧИВАНИЕ МОДЕЛЕЙ ---
echo "===== ШАГ 4: Скачивание AI-моделей... ====="
if [[ -n "${HF_TOKEN}" ]]; then
    echo "Обнаружен токен HF_TOKEN. Выполняется вход в Hugging Face CLI..."
    conda run -n $CONDA_ENV_NAME huggingface-cli login --token ${HF_TOKEN}
fi

# Логика скачивания моделей, адаптированная из entrypoint.sh
# --- Скачивание xcodec ---
XCODEC_DIR="${REPO_DIR}/inference/xcodec_mini_infer"
if [ ! -d "$XCODEC_DIR" ]; then
    echo "Скачивание xcodec_mini_infer..."
    conda run -n $CONDA_ENV_NAME huggingface-cli download m-a-p/xcodec_mini_infer --local-dir "$XCODEC_DIR"
else
    echo "Модель xcodec_mini_infer уже скачана."
fi

# --- Скачивание основных моделей ---
declare -A MODELS_BF16
MODELS_BF16["YuE-s1-7B-anneal-en-cot"]="m-a-p/YuE-s1-7B-anneal-en-cot:${MODEL_DIR}/YuE-s1-7B-anneal-en-cot"
MODELS_BF16["YuE-s1-7B-anneal-en-icl"]="m-a-p/YuE-s1-7B-anneal-en-icl:${MODEL_DIR}/YuE-s1-7B-anneal-en-icl"
MODELS_BF16["YuE-s2-1B-general"]="m-a-p/YuE-s2-1B-general:${MODEL_DIR}/YuE-s2-1B-general"
MODELS_BF16["YuE-upsampler"]="m-a-p/YuE-upsampler:${MODEL_DIR}/YuE-upsampler"
# (Для краткости добавлены не все модели, можно расширить по аналогии с entrypoint.sh)

if [ "$DOWNLOAD_MODELS" != "false" ]; then
    echo "Выбрана опция скачивания: $DOWNLOAD_MODELS"
    SELECTED_MODELS=()
    MODELS=()

    if [ "$DOWNLOAD_MODELS" = "all_bf16" ]; then
        SELECTED_MODELS=("${!MODELS_BF16[@]}")
        for key in "${!MODELS_BF16[@]}"; do MODELS[$key]="${MODELS_BF16[$key]}"; done
    # Добавьте здесь elif для all_int8, all_nf4, all по аналогии с entrypoint.sh, если нужно
    else
        IFS=',' read -r -a SELECTED_MODELS <<< "$DOWNLOAD_MODELS"
        for MODEL in "${SELECTED_MODELS[@]}"; do
            if [[ -v MODELS_BF16[$MODEL] ]]; then MODELS[$MODEL]="${MODELS_BF16[$MODEL]}"; fi
        done
    fi

    for MODEL in "${SELECTED_MODELS[@]}"; do
        SOURCE_DEST_STRING=${MODELS[$MODEL]}
        if [[ -z "$SOURCE_DEST_STRING" ]]; then
            echo "ВНИМАНИЕ: Модель '$MODEL' не найдена в списках. Пропускается."
            continue
        fi
        SOURCE=$(echo $SOURCE_DEST_STRING | cut -d':' -f1)
        DESTINATION=$(echo $SOURCE_DEST_STRING | cut -d':' -f2)

        if [ ! -d "$DESTINATION" ]; then
            echo "Скачивание модели: $MODEL из $SOURCE..."
            conda run -n $CONDA_ENV_NAME huggingface-cli download "$SOURCE" --local-dir "$DESTINATION"
        else
            echo "Модель $MODEL уже скачана в $DESTINATION."
        fi
    done
else
    echo "Скачивание моделей пропущено (DOWNLOAD_MODELS=false)."
fi
echo "===== Скачивание моделей завершено. ====="
echo

# --- ШАГ 5: ПАТЧИНГ БИБЛИОТЕКИ TRANSFORMERS ---
#echo "===== ШАГ 5: Патчинг библиотеки 'transformers'... ====="
#SITE_PACKAGES_DIR=$(conda run -n $CONDA_ENV_NAME python -c "import site; print(site.getsitepackages()[0])")
#TRANSFORMERS_PATH="$SITE_PACKAGES_DIR/transformers"

#if [ ! -d "$TRANSFORMERS_PATH" ]; then
    #echo "ОШИБКА: Библиотека 'transformers' не найдена. Установка не удалась."
    #exit 1
#fi

#CUSTOM_MODELING_LLAMA="transformers/models/llama/modeling_llama.py"
#CUSTOM_GENERATION_UTILS="transformers/generation/utils.py"
#TARGET_MODELING_LLAMA="$TRANSFORMERS_PATH/models/llama/modeling_llama.py"
#TARGET_GENERATION_UTILS="$TRANSFORMERS_PATH/generation/utils.py"

#echo "Копирование кастомных файлов..."
#cp -v "$CUSTOM_MODELING_LLAMA" "$TARGET_MODELING_LLAMA"
#cp -v "$CUSTOM_GENERATION_UTILS" "$TARGET_GENERATION_UTILS"
#echo "===== Патчинг успешно завершен. ====="
#echo

# --- ШАГ 6: СОЗДАНИЕ ФАЙЛА С ПЕРЕМЕННЫМИ ОКРУЖЕНИЯ ---
echo "===== ШАГ 6: Создание файла env_vars.sh... ====="
cat << EOF > env_vars.sh
#!/bin/bash
# Активируйте этот файл перед запуском приложения: source env_vars.sh
echo "Установка переменных окружения..."
export REPO_DIR="${REPO_DIR}"
export MODEL_DIR="${MODEL_DIR}"
export PATH="\$REPO_DIR:\$REPO_DIR/inference:\$REPO_DIR/inference/xcodec_mini_infer:\$REPO_DIR/inference/xcodec_mini_infer/descriptaudiocodec:\$PATH"
echo "Переменные окружения установлены. Проверьте \$PATH."
EOF
echo "Файл env_vars.sh создан."
echo

echo "=================================================================="
echo "      🚀 Установка и настройка проекта полностью завершены! 🚀     "
echo "=================================================================="
echo "ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:"
echo
echo "1. Активируйте переменные окружения (нужно делать в КАЖДОЙ новой сессии терминала):"
echo "   source env_vars.sh"
echo
echo "2. Активируйте окружение Conda (если еще не активно):"
echo "   conda activate $CONDA_ENV_NAME"
echo
echo "3. Запустите веб-интерфейс:"
echo "   python inference/interface.py"
echo "------------------------------------------------------------------"