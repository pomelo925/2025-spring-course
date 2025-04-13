FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# GUI
RUN apt update && apt install -y \
    python3.10 python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN python3.10 -m pip install --upgrade pip && \
    pip install \
    numpy==1.24.4 \
    matplotlib \
    opencv-python \
    cloudpickle

# GPU acceleration (CUDA 12.8)
RUN pip3 install --pre torch \
    torchvision \
    torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
CMD [ "/bin/bash" ] 