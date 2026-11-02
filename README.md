# NeuroStream.IO

Local proof-of-concept of a fault-tolerant medical telemetry -> ML inference pipeline.

```
simulate_patient.py --HTTP--> NestJS gateway --> Kafka (neuro-raw-signals)
                                                    |
                                                    v
                         Python worker (NumPy/SciPy: sliding-window z-score + FFT band powers,
                                        dynamic batching, at-least-once offsets, DLQ)
                                                    | gRPC (proto/inference.proto)
                                                    v
                         ML server (1-D CNN trained on real Bonn EEG, tuned threshold)
                                                    |
                                                    v
                                      worker logs CRITICAL ALERT on seizure
```

## Layout
```
proto/inference.proto            shared gRPC contract
docker-compose.yml               Kafka (KRaft), topic init, ml-server, gateway, worker
services/ingestion-gateway/      NestJS + class-validator + @nestjs/microservices Kafka
services/worker/                 Kafka consumer, preprocessing, batcher, gRPC client, alerting (+ tests)
services/ml-server/              gRPC inference server
simulator/simulate_patient.py    synthetic EEG-like stream with random spikes
```

## Run
```bash
docker compose up -d --build        # first build downloads torch (several minutes); needs models/ from training first
pip install -r simulator/requirements.txt
python simulator/simulate_patient.py --spike-prob 0.3
```

## Train the model (real EEG, real neural network)
```bash
python -m venv .venv && .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install numpy scipy pandas grpcio grpcio-tools protobuf requests pytest
# download Bonn EEG sets z,o,n,f,s from ukbonn.de into training/data/<set>/ (see training/train.py header)
.venv/Scripts/python training/train.py     # writes models/spike_cnn.pt, meta.json, heldout_test_files.json
```
Held-out test result (75 unseen recordings, 1725 one-second windows): accuracy 97.6%, recall 95.1%, precision 93.2%, AUC 0.998.

## No-Docker demo (replays held-out real EEG)
```bash
.venv/Scripts/python demo/run_demo.py --chunks 40
```
Real preprocessing, batching, gRPC client/server, trained CNN and alerting; Kafka and the NestJS gateway are replaced by an in-memory queue and a validation function.

## Verify (three one-liners)
```bash
# 1. Gateway accepts and publishes telemetry
docker compose logs -f ingestion-gateway | grep published
# 2. Messages are really in Kafka
docker compose exec kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic neuro-raw-signals --from-beginning --max-messages 1 | head -c 300
# 3. Worker -> ML server -> closed-loop alert
docker compose logs -f worker | grep -E "CRITICAL ALERT|anomaly="
```
(Windows PowerShell: replace `grep X` with `Select-String X`.) Run the simulator in a separate terminal while these run.

## Tests
```bash
cd services/worker && pip install -r requirements.txt && python -m pytest tests -q   # needs PYTHONPATH=. (set automatically via conftest)
cd services/ingestion-gateway && npm install && npm test   # 14 tests: DTO rules + real HTTP e2e against the actual app with a fake Kafka producer
```

## Design notes / honest caveats
- **Delivery:** at-least-once. Offsets commit only after inference succeeds; if the ML server stays down past the retry budget the worker rewinds and retries. Malformed messages go to `neuro-dead-letter`.
- **Model:** a 1-D CNN trained from scratch on the Bonn EEG benchmark (seizure vs. non-seizure, split by recording). That benchmark is *easy* (published models reach ~99%) and uses clean, single-channel, 1-second windows; it is NOT evidence of clinical performance on real hospital streams (noise, many patients, many channels, much rarer seizures).
- Single-node Kafka, plaintext, no auth: laptop demo only.
