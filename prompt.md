Act as a Principal ML Systems Architect and Senior Backend Engineer. I am a mid/senior software engineer with 4.5+ years of full-stack/backend experience (specializing in NestJS, TypeScript, Kafka, and Python). I want to build a functional, local proof-of-concept project called "NeuroStream-Inference" to showcase my ability to build high-performance distributed systems that host, serve, and orchestrate deep learning models for the medical world.

This project must run locally on a single laptop but be structured as a decoupled, production-grade microservices architecture. It needs to demonstrate strict software engineering principles (clean code, error handling, decoupled architecture) and bridge backend systems with data research.

### Project Overview:

A real-time, fault-tolerant simulation of a medical data pipeline. A telemetry script simulates a hospital machine streaming neurological/physiological signals. A NestJS ingestion gateway receives them, pushes them to Apache Kafka, a Python worker pulls them from Kafka, applies signal preprocessing (NumPy/SciPy), and sends optimized requests via gRPC to an isolated machine learning inference server (TorchServe or NVIDIA Triton running in Docker) hosting a pre-trained biomedical model.

### Guidelines for Generation:

Generate a complete, step-by-step technical execution blueprint and the exact boilerplate code required to get this running locally. Organize the output cleanly, specifying folder structures and file names so I can copy-paste them directly into my IDE.

Follow this execution plan:

#### 1. Architectural File Structure

Provide a clean, unified project directory layout separating the NestJS gateway, the Python worker, the docker-compose configurations, and the data simulation script.

#### 2. Infrastructure Setup (Docker Compose)

Provide a `docker-compose.yml` file that spins up:

- Apache Kafka (using a single-node KRaft broker like `confluentinc/cp-kafka` to keep it lightweight for a laptop).
- An ML Model Serving container (use TorchServe or Python-driven FastAPI/Uvicorn optimized as a clean inference server if Triton is too heavy for a basic local laptop setup. Let's stick to a highly optimized Python-gRPC or FastAPI serving container running a real pre-trained model like a simple ResNet or a Hugging Face bio-signal classifier mapped to mock dimensions).

#### 3. Ingestion Service (NestJS & TypeScript)

Create a clean NestJS application with a single high-throughput POST endpoint `/api/v1/telemetry/stream`.

- Configured to accept payload chunks: `{"patientId": "string", "timestamp": number, "signalVector": number[]}`.
- Implements the native NestJS `@nestjs/microservices` Kafka module.
- Validates inputs using `class-validator` and publishes raw data directly to a Kafka topic named `neuro-raw-signals`.

#### 4. Preprocessing & ML Orchestration Worker (Python)

Create a Python worker script utilizing `kafka-python` or `confluent-kafka`.

- Subscribes to `neuro-raw-signals`.
- Uses NumPy/SciPy to perform a real-world signal processing step (e.g., a simple Fast Fourier Transform (FFT) or sliding-window normalization to demonstrate matrix manipulation).
- Batches incoming events dynamically and forwards them via low-latency gRPC or optimized HTTP/2 to the ML Inference server.

#### 5. ML Model Server Setup (Python/TorchServe Layer)

Provide the script that wraps an existing, lightweight pre-trained model (e.g., an anomaly detection model or an image classification model adapted to act as an EEG spike detector).

- The server processes the numerical arrays, runs inference, and returns a classification output (e.g., `{"anomalyDetected": true, "confidence": 0.94}`).
- When an anomaly is detected, the Python worker must catch the response and log it as a simulated critical notification payload to show a closed-loop system.

#### 6. Medical Telemetry Simulator Script (Python)

Provide a simple utility script `simulate_patient.py` that uses NumPy to generate a synthetic mathematical wave (e.g., a sine wave with occasional random high-amplitude spikes simulating anomalous neurological activity) and streams it continuously via HTTP POST requests to the NestJS endpoint.

#### 7. Verification Guide

Provide 3 single-line terminal commands I can run to prove the end-to-end flow is fully working, showing logs of the stream flowing from the simulator -> NestJS -> Kafka -> Python Worker -> ML Server -> Anomaly Alert.

Make the code production-grade, highly readable, and focused on showcasing architectural patterns that will impress engineering managers at top health-tech companies. Let's begin step by step.
