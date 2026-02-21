# AIRI Assistant

AIRI Assistant is a lightweight, edge-optimized voice assistant designed for real-time wake word detection and offline speech synthesis. It uses custom machine learning models trained for keyword spotting and runs efficiently on resource-constrained systems with minimal latency and memory usage.

The assistant is designed with a privacy-first approach, ensuring core functionality runs locally without requiring cloud processing.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Wake Word Model Development](#wake-word-model-development)
- [Dataset Preparation](#dataset-preparation)
- [Dataset Variants](#dataset-variants)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Model Testing and Validation](#model-testing-and-validation)
- [Deployment](#deployment)
- [Text-to-Speech](#text-to-speech)
- [Model Optimization](#model-optimization)
- [Supported Platforms](#supported-platforms)
- [Use Cases](#use-cases)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## Features

- Custom wake word detection ("Hello Airi")
- Fully offline inference
- Optimized INT8 quantized machine learning models
- Synthetic and real voice training support
- Offline text-to-speech synthesis
- Docker-based Linux deployment
- Low CPU and memory usage
- Modular and extensible architecture

---

## System Architecture
