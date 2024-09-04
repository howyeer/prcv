FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ARG MODEL="yolo_world_v2_l_vlpan_bn_sgd__prcv2.py"
ARG WEIGHT="prcv2.pth"

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip     \
    libgl1-mesa-glx \
    libsm6          \
    libxext6        \
    libxrender-dev  \
    libglib2.0-0    \
    git             \
    python3-dev     \
    python3-wheel

RUN pip3 install --upgrade pip \
    && pip3 install   \
        gradio        \
        opencv-python \
        supervision   \
        mmengine      \
        setuptools    \
        openmim       \
    && mim install mmcv==2.0.0 \
    && pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        wheel         \
        torch         \
        torchvision   \
        torchaudio

COPY . /prcv
WORKDIR /prcv

RUN pip3 install -e .

RUN curl -o weights/$WEIGHT -L https://huggingface.co/howyeer/prcv/blob/main/prcv2.pth

ENTRYPOINT [ "python3", "demo.py", "prcv_infer.py"]
CMD ["configs/finetune_coco/$MODEL", "weights/$WEIGHT"]