ARG BASE_TAG=main
FROM pytorch/pytorch:latest

ENV HF_DATASETS_CACHE="/workspace/data/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/workspace/data/huggingface-cache/hub"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server tmux git nano  && \
    rm -rf /var/lib/apt/lists/*

# Create the .ssh directory, set its permissions, and add the public key to authorized_keys

# Clone the repository
RUN git clone https://github.com/Decycle/fine-tune-experiments.git /root/fine-tune-experiments

WORKDIR /root/fine-tune-experiments/axolotl
RUN pip3 install -e .
RUN pip3 install -U git+https://github.com/huggingface/peft.git
RUN pip3 install langchain openai fastapi uvicorn[standard]

COPY start.sh /start.sh

WORKDIR /workspace
# Start the SSH service and sleep indefinitely
CMD ["/start.sh"]