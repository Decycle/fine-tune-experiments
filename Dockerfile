ARG BASE_TAG=main
FROM winglian/axolotl:main-py3.9-cu118-2.0.1

ENV HF_DATASETS_CACHE="/workspace/data/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/workspace/data/huggingface-cache/hub"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server tmux git nano  && \
    rm -rf /var/lib/apt/lists/*

# Create the .ssh directory, set its permissions, and add the public key to authorized_keys

RUN mkdir -p ~/.ssh && \
    chmod 700 ~/.ssh && \
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys

# Clone the repository
RUN git clone https://github.com/Decycle/fine-tune-experiments.git /workspace/fine-tune-experiments

WORKDIR /workspace/fine-tune-experiments/axolotl
RUN pip3 install -e .

# Install python packages
WORKDIR /workspace/fine-tune-experiments
RUN pip3 install langchain openai fastapi uvicorn[standard]

# Copy the .env file from the host to the Docker image
COPY .env /workspace/fine-tune-experiments/.env
# Start the SSH service and sleep indefinitely
CMD service ssh start && sleep infinity