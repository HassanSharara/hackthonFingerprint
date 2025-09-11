FROM debian:bookworm

WORKDIR /usr/src/myapp

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y sudo wget unzip libopencv-dev ninja-build clang curl llvm-dev libclang-dev build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
    # Install Rust (via rustup)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN apt-get update
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
RUN unzip libtorch-shared-with-deps-latest.zip -d /opt/libtorch
ENV LIBTORCH=/opt/libtorch
ENV LIBTORCH_USE_CUDA=0
COPY Cargo.toml .
COPY src ./src

CMD ["bash"]
