# Multi-Node PyTorch Distributed Training on GCP Vertex AI

This repository implements multi-node PyTorch Distributed Data Parallel (DDP) training using NVIDIA L4 GPUs on Google Cloud Vertex AI with Spot VMs.

## Project Structure
- `trainer/task.py`: PyTorch DDP training script with automatic Vertex AI `CLUSTER_SPEC` discovery.
- `pyproject.toml`: Package definition and dependencies (managed with `uv`).
- `launch_job.py`: Vertex AI launcher targeting `g2-standard-8` (Spot instances).

## Quickstart Guide

### 1. Local Setup
```bash
uv sync