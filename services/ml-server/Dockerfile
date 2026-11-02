FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY services/ml-server/requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY proto/ ./proto/
COPY services/ml-server/app ./app
RUN python -m grpc_tools.protoc -I proto --python_out=app --grpc_python_out=app proto/inference.proto \
    && sed -i 's/^import inference_pb2/from app import inference_pb2/' app/inference_pb2_grpc.py

# Trained weights + tuned threshold (produced by training/train.py)
COPY models/ ./models/
ENV MODEL_DIR=/srv/models

EXPOSE 50051
CMD ["python", "-m", "app.server"]
