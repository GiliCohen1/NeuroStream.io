FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY services/worker/requirements.txt .
RUN pip install -r requirements.txt

COPY proto/ ./proto/
COPY services/worker/worker ./worker
RUN python -m grpc_tools.protoc -I proto --python_out=worker --grpc_python_out=worker proto/inference.proto \
    && sed -i 's/^import inference_pb2/from worker import inference_pb2/' worker/inference_pb2_grpc.py

CMD ["python", "-m", "worker.main"]
