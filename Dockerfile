FROM lmsysorg/sglang:v0.5.14-cu129

# Set working directory to the one already used by the base image
WORKDIR /sgl-workspace

# install dependencies (pip is already available in the base image)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source files
COPY handler.py engine.py utils.py download_model.py test_input.json ./
COPY public/ ./public/

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    SGLANG_VLM_CACHE_SIZE_MB=0

# Model download script execution
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 download_model.py; \
    fi

# Create file storage directory for VLM image uploads
RUN mkdir -p ${BASE_PATH}/file-storage

# HEALTHCHECK: Docker monitors sglang server health
# Interval: 30s, Timeout: 10s, Retries: 3 (fails after 90s of no response)
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:30000/v1/models', timeout=5)" || exit 1

CMD ["python3", "handler.py"]