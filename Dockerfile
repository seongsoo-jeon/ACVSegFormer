FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    bzip2 \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
RUN pip install pandas \
    timm \
    resampy \
    soundfile

WORKDIR /app/AVSegFormer
COPY ./ ./

# wondow 시에 
RUN find . -name "*.sh" -exec dos2unix {} +

# 'IndexError'를 해결하기 위해 타겟 CUDA 아키텍처를 명시적으로 설정
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
# setup.py가 빌드 중 GPU 가시성 없이 CUDA를 강제로 사용하도록 환경 변수 설정
ENV FORCE_CUDA=1

WORKDIR /app/AVSegFormer/ops


RUN sh make.sh
RUN mkdir -p /app/AVSegFormer/data
WORKDIR /app/AVSegFormer

CMD ["/bin/bash"]