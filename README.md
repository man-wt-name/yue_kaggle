<div align="center">
   <h1>YuE Interface</h1>
    <picture>
        <source srcset="./assets/logo/黑底.svg" media="(prefers-color-scheme: dark)">
        <img src="./assets/logo/白底.svg" width="25%">
    </picture>
</div>

Welcome to the **YuE Interface**, a robust and user-friendly Docker-based solution for generating music using YuE models. This interface leverages Gradio for an interactive web UI, enabling you to configure and execute music generation tasks seamlessly. Whether you're running locally with Docker or deploying on RunPod, this guide will help you get started.

> **Note**: This project is a fork of the official [YuE repository](https://github.com/multimodal-art-projection/YuE). For more information about inference settings or tips see the official repository README. 

> **Note**: There is this other fork that I made (from another fork made by sgsdxzy hehehe) for a more optimized version of YuE that uses exllamav2, the generated music has a little degradation but the inference is extremely faster than what you will get in this fork made from the official repository. If you want maximum quality it is recommended to use this fork here, otherwise if your focus is speed and you don't care much about the degradation problems coming from quantization feel free to use this [fork](https://github.com/alisson-anjos/YuE-exllamav2-UI)

## Interface

![preview gradio](/preview.png)

## News and Updates

> **Note**: Keep an eye on updates to know when you need to do a docker pull alissonpereiraanjos/yue-interface:latest

* **2025.02.05 (Oficial Repository) 🔥**: Added repetition penalty
* **2025.02.03 🔥**: Added Optimizations (MMGP - Memory Management for the GPU Poor by DeepBeepMeep)
* **2025.02.01 (Oficial Repository) 🧑🏻‍💻 Inference Update**: 1. Support dual-track ICL mode. 2. Fix "instrumental" naming bug in output files. 3. Support seeding. 
* **2025.01.31 🔥**: Added possibility to select existing models using a dropdown, added Refresh File Explorer button and Warnings are ignored in the log.
* **2025.01.30 🔥**: Added support for NF4 quantized models. 
* **2025.01.30 🔥**: Added the seed field and also improvements in the genres field.
* **2025.01.29 🔥**: Added support for INT8 quantized models. 

## Features

- **Docker Image**: Pre-configured Docker image for easy deployment.
- **Web UI (Gradio)**: Intuitive interface for configuring and executing music generation.
- **NVIDIA GPU Support**: Mandatory support for NVIDIA GPUs to accelerate processing.
- **Model Management**: Ability to download all or specific YuE models based on your needs.
- **Volume Mapping**: Map model and output directories from the host system into the container.
- **Real-time Logging**: Monitor generation logs directly from the web interface.
- **Audio Playback and Download**: Listen to generated audio and download it directly from the interface.

## Prerequisites

### Docker

Ensure you have Docker installed on your system. Follow the official Docker installation guides for your platform:

- [Install Docker on Windows](https://docs.docker.com/desktop/windows/install/)
- [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

### NVIDIA GPU Support

This interface **requires NVIDIA GPUs** for acceleration. Ensure you have the necessary hardware and drivers set up.

1. **Linux**:
   - Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
   - Ensure your NVIDIA drivers are properly installed and up to date.

2. **Windows/macOS**:
   - Refer to the respective Docker and NVIDIA documentation for GPU passthrough (e.g., WSL2 on Windows).
   - **Note**: GPU support is mandatory. Without compatible NVIDIA GPUs, the container will not function correctly.

## Using Docker Compose

To simplify the setup and management of the YuE Interface, you can use Docker Compose. Docker Compose allows you to define and run multi-container Docker applications with a single configuration file (`docker-compose.yml`). Below are the steps to get started.

> **Note**: This **docker-compose.yml** file already exists in the root of the repository, you just need to download or copy the file and replace the directory mapping and run the command in the same directory as the file, see the explanation below.

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  yue-interface:
    image: alissonpereiraanjos/yue-interface:latest  # Docker image for YuE Interface
    container_name: yue-interface  # Name of the container
    restart: unless-stopped  # Restart policy: always restart unless manually stopped
    ports:
      - "7860:7860"  # Map port 7860 (Gradio UI) to the host
      - "8888:8888"  # Optional: Map an additional port (JupyterLab)
    environment:
      - DOWNLOAD_MODELS=all  # Download all models. Replace "all" with specific model keys if needed.
                             # Example: DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot
    volumes:
      - /path/to/models:/workspace/models  # Map the host's model directory to the container
      - /path/to/outputs:/workspace/outputs  # Map the host's output directory to the container
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # Enable GPU support (requires NVIDIA GPU and drivers)
```

### Explanation of the Configuration

- **`image`**: Specifies the Docker image to use (`alissonpereiraanjos/yue-interface:latest`).
- **`container_name`**: Sets a name for the container (`yue-interface`).
- **`restart: unless-stopped`**: Ensures the container restarts automatically unless manually stopped.
- **`ports`**: Maps container ports to the host:
  - `7860:7860`: Port for accessing the Gradio UI.
  - `8888:8888`: Optional additional port (JupyterLab).
- **`environment`**: Defines environment variables:
  - `DOWNLOAD_MODELS=all`: Downloads all available models. Replace `all` with specific model keys (e.g., `YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot`) to download only selected models.
- **`volumes`**: Maps host directories to the container:
  - `/path/to/models:/workspace/models`: Directory where models will be stored.
  - `/path/to/outputs:/workspace/outputs`: Directory where generated outputs will be saved.
  - Replace `/path/to/models` and `/path/to/outputs` with the actual paths on your system.
- **`deploy.resources.reservations.devices`**: Enables GPU support in the container (requires NVIDIA GPU and drivers).

### How to Use Docker Compose

**Run Docker Compose**: Navigate to the directory where the `docker-compose.yml` file is saved and run:

   ```bash
   docker-compose up -d
   ```

   The `-d` flag starts the container in detached mode (background).

4. **Access the Interface**: Once the container is running, access the Gradio UI at `http://localhost:7860`.

### Useful Commands

- **Stop the container**:

  ```bash
  docker-compose down
  ```

- **View logs**:

  ```bash
  docker-compose logs -f
  ```

- **Update the image**:

  ```bash
  docker-compose pull
  docker-compose up -d
  ```

### Customization

- **Specific models**: To download only specific models, modify the `DOWNLOAD_MODELS` environment variable in the `docker-compose.yml` file. For example:

  ```yaml
  environment:
    - DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot
  ```

- **Different ports**: If you need to use different ports, adjust the port mappings under the `ports` section.

### Notes

- Ensure the **NVIDIA Container Toolkit** is installed and properly configured on your system for GPU support.
- If you're using Windows, replace the volume paths with Windows-style paths, such as `D:\AI\yue\models:/workspace/models`.

With Docker Compose, you can easily manage and deploy the YuE Interface with minimal setup! 🚀

## How to Run using only Docker

## Docker Image

The YuE Interface Docker image is hosted on Docker Hub:

```
alissonpereiraanjos/yue-interface:latest
```

## Environment Variables

- **DOWNLOAD_MODELS**: Determines which models to download.
  - Set to `all` to download all available models (BF16, INT8).
  - Set to `all_bf16` to download all available models (BF16).
  - Set to `all_int8` to download all available models (INT8).
  - Set to `all_nf4` to download all available models (NF4).
  - Alternatively, specify a comma-separated list of model keys to download specific models (e.g., `DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot`).

### You can also not use docker compose and do everything manually just using docker.

### Basic Run Command

To start the YuE Interface with all models downloaded and NVIDIA GPU support enabled:

```bash
docker run --gpus all -d \
  -p 7860:7860 \
  -p 8888:8888 \
  -e DOWNLOAD_MODELS=all \
  alissonpereiraanjos/yue-interface:latest
```

- `--gpus all`: Enables NVIDIA GPU support.
- `-d`: Runs the container in detached mode (background).
- `-p 7860:7860`: Exposes port `7860` for accessing the Gradio UI at [http://localhost:7860](http://localhost:7860).
- `-p 8888:8888`: Exposes port `8888` for additional services if applicable.
- `-e DOWNLOAD_MODELS=all`: Downloads all available models upon initialization.

### Specifying Models to Download

To download specific models, set the `DOWNLOAD_MODELS` environment variable to a comma-separated list of model keys:

```bash
docker run --gpus all -d \
  -p 7860:7860 \
  -p 8888:8888 \
  -e DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot \
  alissonpereiraanjos/yue-interface:latest
```

### Mapping Directories for Models and Output

You can mount host directories to store models and outputs outside the container:

```bash
docker run --gpus all -it \
  -v /path/to/models:/workspace/models \
  -v /path/to/outputs:/workspace/outputs \
  -p 7860:7860 \
  -p 8888:8888 \
  -e DOWNLOAD_MODELS=false \
  alissonpereiraanjos/yue-interface:latest
```

- `-v /path/to/models:/workspace/models`: Mounts the host's `/path/to/models` directory to `/workspace/models` inside the container.
- `-v /path/to/outputs:/workspace/outputs`: Mounts the host's `/path/to/outputs` directory to `/workspace/outputs` inside the container.
- `-e DOWNLOAD_MODELS=false`: Skips automatic model downloads (useful if models are already present in the mounted directories).

#### Example for Windows:

```bash
docker run --gpus all -it \
  -v D:\AI\yue\models:/workspace/models \
  -v D:\AI\yue\outputs:/workspace/outputs \
  -p 7860:7860 \
  -p 8888:8888 \
  -e DOWNLOAD_MODELS=false \
  alissonpereiraanjos/yue-interface:latest
```

### Mapping Gradio to Port 8888

If you prefer to map Gradio to port `8888` in addition to the default `7860`, adjust the port mapping accordingly:

```bash
docker run --gpus all -d \
  -p 7860:7860 \
  -p 8888:8888 \
  -e DOWNLOAD_MODELS=all \
  alissonpereiraanjos/yue-interface:latest
```

In this example, the Gradio UI inside the container is accessible on both ports `7860` and `8888`.

## Summary of Options

- `-v /host/path:/container/path`: Mount host directories into the container.
- `-p host_port:container_port`: Map container ports to host ports.
  - Example: `-p 7860:7860` maps the container's `7860` port to host's `7860`.
  - Example: `-p 8888:8888` maps the container's `8888` port to host's `8888`.
- `-e VARIABLE=value`: Set environment variables.
  - Example: `-e DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot`
- `--gpus all`: Enables NVIDIA GPU support.
- `-it`: Start in interactive mode (useful for debugging).
- `-d`: Start in detached mode (runs in the background).

Use these options to tailor the setup to your environment and requirements.

## Available Models

Below is the list of available YuE models that you can download by specifying their keys in the `DOWNLOAD_MODELS` environment variable:

| Model Key                         | Model HF Repository                             | Container Path                                      | Quantization |
|-----------------------------------|----------------------------------------------|----------------------------------------------------|--------------|
| `xcodec_mini_infer`               | [`m-a-p/xcodec_mini_infer`](https://huggingface.co/m-a-p/xcodec_mini_infer) | `/workspace/YuE-Interface/inference/xcodec_mini_infer` | N/A          |
| `YuE-s1-7B-anneal-en-cot`         | [`m-a-p/YuE-s1-7B-anneal-en-cot`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot) | `/workspace/models/YuE-s1-7B-anneal-en-cot`       | BF16         |
| `YuE-s1-7B-anneal-en-icl`         | [`m-a-p/YuE-s1-7B-anneal-en-icl`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl) | `/workspace/models/YuE-s1-7B-anneal-en-icl`       | BF16         |
| `YuE-s1-7B-anneal-jp-kr-cot`      | [`m-a-p/YuE-s1-7B-anneal-jp-kr-cot`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-cot) | `/workspace/models/YuE-s1-7B-anneal-jp-kr-cot`    | BF16         |
| `YuE-s1-7B-anneal-jp-kr-icl`      | [`m-a-p/YuE-s1-7B-anneal-jp-kr-icl`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-icl) | `/workspace/models/YuE-s1-7B-anneal-jp-kr-icl`    | BF16         |
| `YuE-s1-7B-anneal-zh-cot`         | [`m-a-p/YuE-s1-7B-anneal-zh-cot`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot) | `/workspace/models/YuE-s1-7B-anneal-zh-cot`       | BF16         |
| `YuE-s1-7B-anneal-zh-icl`         | [`m-a-p/YuE-s1-7B-anneal-zh-icl`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-icl) | `/workspace/models/YuE-s1-7B-anneal-zh-icl`       | BF16         |
| `YuE-s2-1B-general`               | [`m-a-p/YuE-s2-1B-general`](https://huggingface.co/m-a-p/YuE-s2-1B-general) | `/workspace/models/YuE-s2-1B-general`             | BF16         |
| `YuE-upsampler`                   | [`m-a-p/YuE-upsampler`](https://huggingface.co/m-a-p/YuE-upsampler) | `/workspace/models/YuE-upsampler`                 | BF16         |
| `YuE-s1-7B-anneal-en-cot-int8`    | [`Alissonerdx/YuE-s1-7B-anneal-en-cot-int8`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-en-cot-int8) | `/workspace/models/YuE-s1-7B-anneal-en-cot-int8`  | INT8         |
| `YuE-s1-7B-anneal-en-icl-int8`    | [`Alissonerdx/YuE-s1-7B-anneal-en-icl-int8`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-en-icl-int8) | `/workspace/models/YuE-s1-7B-anneal-en-icl-int8`  | INT8         |
| `YuE-s1-7B-anneal-jp-kr-cot-int8` | [`Alissonerdx/YuE-s1-7B-anneal-jp-kr-cot-int8`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-jp-kr-cot-int8) | `/workspace/models/YuE-s1-7B-anneal-jp-kr-cot-int8` | INT8         |
| `YuE-s1-7B-anneal-jp-kr-icl-int8` | [`Alissonerdx/YuE-s1-7B-anneal-jp-kr-icl-int8`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-jp-kr-icl-int8) | `/workspace/models/YuE-s1-7B-anneal-jp-kr-icl-int8` | INT8         |
| `YuE-s1-7B-anneal-zh-cot-int8`    | [`Alissonerdx/YuE-s1-7B-anneal-zh-cot-int8`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-zh-cot-int8) | `/workspace/models/YuE-s1-7B-anneal-zh-cot-int8`  | INT8         |
| `YuE-s1-7B-anneal-zh-icl-int8`    | [`Alissonerdx/YuE-s1-7B-anneal-zh-icl-int8`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-zh-icl-int8) | `/workspace/models/YuE-s1-7B-anneal-zh-icl-int8`  | INT8         |
| `YuE-s2-1B-general-int8`          | [`Alissonerdx/YuE-s2-1B-general-int8`](https://huggingface.co/Alissonerdx/YuE-s2-1B-general-int8) | `/workspace/models/YuE-s2-1B-general-int8`        | INT8         |
| `YuE-s1-7B-anneal-en-cot-nf4`     | [`Alissonerdx/YuE-s1-7B-anneal-en-cot-nf4`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-en-cot-nf4) | `/workspace/models/YuE-s1-7B-anneal-en-cot-nf4`   | NF4          |
| `YuE-s1-7B-anneal-en-icl-nf4`     | [`Alissonerdx/YuE-s1-7B-anneal-en-icl-nf4`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-en-icl-nf4) | `/workspace/models/YuE-s1-7B-anneal-en-icl-nf4`   | NF4          |
| `YuE-s1-7B-anneal-jp-kr-cot-nf4`  | [`Alissonerdx/YuE-s1-7B-anneal-jp-kr-cot-nf4`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-jp-kr-cot-nf4) | `/workspace/models/YuE-s1-7B-anneal-jp-kr-cot-nf4` | NF4          |
| `YuE-s1-7B-anneal-jp-kr-icl-nf4`  | [`Alissonerdx/YuE-s1-7B-anneal-jp-kr-icl-nf4`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-jp-kr-icl-nf4) | `/workspace/models/YuE-s1-7B-anneal-jp-kr-icl-nf4` | NF4          |
| `YuE-s1-7B-anneal-zh-cot-nf4`     | [`Alissonerdx/YuE-s1-7B-anneal-zh-cot-nf4`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-zh-cot-nf4) | `/workspace/models/YuE-s1-7B-anneal-zh-cot-nf4`   | NF4          |
| `YuE-s1-7B-anneal-zh-icl-nf4`     | [`Alissonerdx/YuE-s1-7B-anneal-zh-icl-nf4`](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-zh-icl-nf4) | `/workspace/models/YuE-s1-7B-anneal-zh-icl-nf4`   | NF4          |



### Model Suffixes Explained

The suffixes in the model keys indicate specific training or optimization techniques applied to the models:

| Suffix | Meaning               | Description                                                                                     |
|--------|-----------------------|-------------------------------------------------------------------------------------------------|
| `COT`  | **Chain-of-Thought**  | Models trained with *Chain-of-Thought* to enhance reasoning and logical generation capabilities.|
| `ICL`  | **In-Context Learning** | Models optimized for *In-Context Learning*, allowing dynamic adaptation based on the provided context.|

**Examples:**

- `YuE-s1-7B-anneal-en-cot`: A model trained with *Chain-of-Thought* techniques.
- `YuE-s1-7B-anneal-en-icl`: A model optimized for *In-Context Learning*.

These suffixes help users identify the specific capabilities and optimizations of each model variant.
### Example: Downloading Specific Models

To download only `YuE-s2-1B-general` and `YuE-s1-7B-anneal-en-cot` models:

```bash
docker run --gpus all -d \
  -p 7860:7860 \
  -p 8888:8888 \
  -e DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot \
  alissonpereiraanjos/yue-interface:latest
```

## Update Docker Image (Important)

To update the Docker image with the latest changes, run:

```bash
docker pull alissonpereiraanjos/yue-interface:latest
```

**Note**: Always update the image before running the container to ensure you have the latest features and fixes. This is especially important when deploying on RunPod, as it pulls the latest image upon creating a new pod.




## Running on RunPod

If you prefer to use **RunPod**, you can quickly deploy an instance based on this image by using the following template link:

[**Deploy on RunPod**](https://runpod.io/console/deploy?template=s8yr44w7br&ref=8t518hht)

This link directs you to the RunPod console, allowing you to set up a machine directly with the YuE Interface image. Configure your GPU, volume mounts, and environment variables as needed.

**Tip**: If you generate music frequently, consider creating a **Network Volume** in RunPod. This allows you to store models and data persistently, avoiding repeated downloads and saving time.

### Example Command for RunPod

```bash
docker run --gpus all -d \
  -v /mnt/models:/workspace/models \
  -v /mnt/outputs:/workspace/outputs \
  -p 7860:7860 \
  -p 8888:8888 \
  -e DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot \
  alissonpereiraanjos/yue-interface:latest
```

Replace `/mnt/models` and `/mnt/outputs` with your desired volume mount paths in RunPod.

## Accessing the Interface

Once the container is running, access the Gradio web UI at:

```
http://localhost:7860
```

If deployed on RunPod, use the provided RunPod URL to access the interface.

## Support

If you encounter any issues or have questions, please open an issue in the [GitHub repository](https://github.com/alissonpereiraanjos/yue-interface).

For official documentation and updates, refer to the [official YuE repository](https://github.com/multimodal-art-projection/YuE).

## Acknowledgements

A special thank you to the developers of the official [YuE repository](https://github.com/multimodal-art-projection/YuE) for their outstanding work and for making this project possible.

---

**Happy Music Generating! 🎶**


## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@misc{yuan2025yue,
  title={YuE: Open Music Foundation Models for Full-Song Generation},
  author={Ruibin Yuan and Hanfeng Lin and Shawn Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Xingjian Du and Xeron Du and Zhen Ye and Tianyu Zheng and Yinghao Ma and Minghao Liu and Lijun Yu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Tianhao Shen and Ziyang Ma and Shangda Wu and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaohuan Zhou and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Yiming Liang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Stephen Huang and Wei Xue and Xu Tan and Yike Guo}, 
  howpublished={\url{https://github.com/multimodal-art-projection/YuE}},
  year={2025},
  note={GitHub repository}
}
```
<br>
