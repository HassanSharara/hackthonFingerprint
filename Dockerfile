FROM python:3.12.4-slim

# Install system dependencies for OpenCV and Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN pip install --upgrade pip

# Install Django + OpenCV from PyPI
RUN pip install django opencv-python scikit-image matplotlib

# Install PyTorch + torchvision + torchaudio from PyTorch CPU wheels
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Set working directory
WORKDIR /app

COPY . /app

# Default command
CMD ["/bin/bash"]
