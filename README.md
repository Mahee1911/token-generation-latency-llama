# Token Generation Latency Benchmarking in LLaMA

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![LLM](https://img.shields.io/badge/LLM-LLaMA-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## Overview

This project investigates **token-generation latency in LLaMA-style Large Language Models (LLMs)** and analyzes the architectural factors that influence inference performance.

While many LLM benchmarks focus on **throughput (tokens per second)**, real-world applications such as chat assistants depend heavily on **latency between tokens**. Token latency directly affects how responsive an AI system feels to users.

This project introduces a **benchmarking and profiling framework** to measure token generation latency during LLaMA inference and identify system-level bottlenecks.

---

## Motivation

Large Language Models generate responses **one token at a time** using autoregressive decoding.

In interactive systems such as:

* Chat assistants
* AI coding tools
* Real-time copilots

users expect **fast and continuous output**.

Two latency metrics are particularly important.

### Time To First Token (TTFT)

The time between sending a request and receiving the **first generated token**.

### Per-Token Latency

The time required to generate **each subsequent token** after the first token.

Reducing these latencies significantly improves **perceived responsiveness** of AI systems.

---

## Project Objectives

The main goals of this project are:

* Design a **repeatable benchmarking framework** for LLaMA inference
* Measure **token-generation latency under controlled conditions**
* Decompose inference latency into **individual transformer components**
* Study how latency **scales with sequence length and model size**
* Identify **hardware and architectural bottlenecks**
* Propose **system-level optimizations** to improve performance

---

## Methodology

### Experimental Setup

Benchmarking experiments are conducted with the following conditions:

* **Batch size:** 1 (simulating interactive chat systems)
* **Multiple prompt lengths tested**
* **Fixed output token length**
* **Warm-up runs executed before measurement**
* **Multiple trials for stable latency estimates**

---

## Metrics Measured

The framework records the following metrics during inference:

* **Time-to-First-Token (TTFT)**
* **Per-token latency**
* **End-to-end response latency**
* **Token throughput (tokens/sec)**

These metrics provide insight into **interactive system performance** rather than just raw throughput.

---

## LLaMA Inference Pipeline

Token generation follows the standard **autoregressive decoding pipeline**:

1. Input prompt processing
2. Token embedding lookup
3. Transformer block execution

   * Self-attention
   * Feed-forward network (MLP)
4. KV-cache read/write operations
5. Logit computation
6. Sampling and next-token selection

This process repeats **for each generated token**.

---

## Latency Decomposition

To identify where latency occurs, the decoding step is decomposed into:

* Embedding lookup
* Attention computation (QKV projections + softmax)
* KV-cache read/write operations
* Feed-forward network (MLP)
* LayerNorm and residual connections
* Sampling / decoding logic
* Framework overhead (kernel launches & synchronization)

This analysis highlights **which components contribute most to token latency**.

---

## Profiling and Analysis

Profiling tools are used to analyze runtime behavior of LLaMA inference.

### Tools Used

* **PyTorch Profiler**
* **Perfetto Timeline Visualization**

Profiling enables detailed observation of:

* Transformer layer execution
* Kernel launches
* Memory operations
* Token generation loops
* End-to-end inference timeline

These traces help identify **performance bottlenecks in the decoding process**.

---

## Scaling Analysis

Latency is evaluated under different configurations:

* Increasing **context length**
* Different **model sizes**
* Different **precision settings**

### Observations

* Per-token latency increases with **longer context lengths**
* KV-cache memory access grows with sequence size
* Memory bandwidth becomes a **primary bottleneck**

---

## Architectural Insights

Key insights from the benchmarking experiments:

* Autoregressive decoding limits parallel execution
* Transformer layers are executed repeatedly for each token
* KV-cache memory access grows with sequence length
* GPU compute units may remain **underutilized at batch size = 1**

As a result, token generation becomes **memory-bandwidth bound rather than compute-bound**.

---

## Optimization Proposal

Based on bottleneck analysis, the project proposes a **KV-cache layout optimization**.

### Idea

* Store KV-cache tensors in **contiguous memory blocks**
* Improve **memory locality**
* Reduce **DRAM memory transactions**
* Increase **L2 cache efficiency**

### Expected Benefits

* Reduced per-token latency
* Improved memory bandwidth utilization
* Better GPU compute efficiency

Estimated improvement:

**~15–20% reduction in steady-state token latency**

---

## Repository Structure

```
token-generation-latency-llama/

analysis/        → Analysis notebooks and scripts  
benchmarks/      → Benchmark execution utilities  
plots/           → Generated performance graphs  
profiling/       → Profiling traces and Perfetto outputs  
results/         → Raw benchmark results  
scripts/         → Execution scripts  

requirements.txt → Python dependencies  
hardware.md      → Hardware configuration  
check.py         → Validation utilities  
check_llama_run.py → Test script for LLaMA inference
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Mahee1911/token-generation-latency-llama.git
cd token-generation-latency-llama
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

Example command to verify LLaMA inference:

```
python check_llama_run.py
```

Benchmark scripts can be executed from the **scripts directory**.

---

## Applications

Understanding token-generation latency is important for:

* Real-time AI chat systems
* Code generation assistants
* Edge AI deployments
* LLM serving infrastructure
* Hardware accelerator design

---

## Learning Outcomes

This project demonstrates how to:

* Build **performance benchmarks for AI systems**
* Profile **deep learning inference pipelines**
* Attribute latency to **architectural components**
* Connect **software behavior with hardware limitations**
* Propose **system-level optimization strategies**

---

## Author

**Mahee Gadhiya**
California State University, Long Beach

Course Project:
**Token Generation Latency Benchmarking in LLaMA**

---

## License

This repository is intended for **academic and research purposes**.
