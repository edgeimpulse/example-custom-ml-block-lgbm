# syntax = docker/dockerfile:experimental
FROM ubuntu:20.04
WORKDIR /scripts

ARG DEBIAN_FRONTEND=noninteractive

# Install base packages
RUN apt update && apt install -y curl zip git lsb-release software-properties-common apt-transport-https vim wget

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-distutils

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py "pip==21.3.1" "setuptools==62.6.0" && \
    rm get-pip.py

# Symlink so we use python3/pip3 calls
RUN rm -f /usr/bin/python3 /usr/bin/pip3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/pip3.10 /usr/bin/pip3

# Install TensorFlow (separate script as this requires a different command on M1 Macs)
COPY dependencies/install_tensorflow.sh install_tensorflow.sh
RUN /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh

# Install jaxlib (separate script as this requires a different command on M1 Macs)
COPY dependencies/install_jaxlib.sh install_jaxlib.sh
RUN /bin/bash install_jaxlib.sh && \
    rm install_jaxlib.sh

RUN apt install -y libgomp1

# Copy other Python requirements in and install them
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip3 install -r requirements.txt

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy the rest of your training scripts in
COPY . ./

ENTRYPOINT [ "python3", "train.py" ]

# FROM public.ecr.aws/g7a8t7v6/jobs-container-keras-export-base:b7c2045436bab937c93d70e93f2658bbc42c88ab
#
# WORKDIR /scripts
#
# # Install extra dependencies here
# COPY requirements.txt ./
# RUN /app/keras/.venv/bin/pip3 install --no-cache-dir -r requirements.txt
#
# # Copy all files to the home directory
# COPY . ./

# The train command (we run this from the keras venv, which has all dependencies)
#ENTRYPOINT [ "./run-python-with-venv.sh", "keras", "train.py" ]
