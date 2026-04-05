FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git wget && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.6 (latest stable with CUDA 12.4 — supports RTX 4090 / Ada Lovelace sm_89)
# PyTorch 2.5.1 also works but 2.6 has better bfloat16 support
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0

# Install training libs
RUN pip3 install --no-cache-dir \
    transformers==4.50.0 \
    peft==0.14.0 \
    datasets==3.2.0 \
    accelerate==1.4.0 \
    bitsandbytes==0.45.3 \
    scipy \
    sentencepiece \
    protobuf

# Copy dataset and training script
WORKDIR /training
COPY train.jsonl valid.jsonl ./
COPY train.py ./

# Run training
CMD ["python3", "train.py"]
