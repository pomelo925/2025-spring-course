FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    numpy>=1.20 \
    matplotlib \
    opencv-python-headless \
    torch \
    cloudpickle

CMD [ "/bin/bash" ]