FROM debian:bookworm-slim

# Install dependencies including Python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp

# Build llama.cpp
RUN mkdir build && cd build && \
    cmake .. -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON && \
    cmake --build . --config Release -j$(nproc)

# Install llama-cpp-python using the built llama.cpp
ENV LLAMA_CPP_LIB=/app/llama.cpp/build/libllama.so
RUN pip3 install llama-cpp-python --break-system-packages --no-cache-dir

# Create directories
RUN mkdir -p /app/models /app/scripts

WORKDIR /app

# Default runs Python
CMD ["python3"]
