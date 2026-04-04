FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git wget && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch + training libs
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    transformers==4.48.0 \
    peft==0.14.0 \
    datasets==3.2.0 \
    accelerate==1.2.1 \
    bitsandbytes==0.45.0 \
    scipy \
    sentencepiece \
    protobuf

# Copy dataset and training script
WORKDIR /training
COPY train.jsonl valid.jsonl ./
COPY train.py ./

# Run training
CMD ["python3", "train.py"]
