# Depth Anything 3 - Benchmark Analysis Report

**Generated:** 2025-12-02 11:48:04

**Model:** Depth Anything 3 Large (da3-large)

**Test Configuration:**
- Images: 5 per test
- Warmup runs: 2
- Benchmark runs: 3-5
- Image size: 504x504

## Executive Summary

**Total Configurations Tested:** 36
**Successful:** 36/36 (100.0%)

### Performance Overview

| Device | Best Throughput | Best Latency | Success Rate |
|--------|----------------|--------------|--------------|
| CUDA | 8.14 img/s | 0.61s | 100% |
| MPS | 1.70 img/s | 2.94s | 100% |
| CPU | 0.54 img/s | 9.31s | 100% |

### Key Findings

1. **Fastest Device:** CUDA (8.14 img/s)
2. **Most Reliable:** All devices achieved 100% success rate
3. **torch.compile:** Disabled on all devices (model incompatibility)
4. **Performance Range:** 15.1x difference between fastest (CUDA) and slowest (CPU)

---

## CUDA Performance Analysis

### Overview

- **Total Configurations:** 16
- **Successful:** 16/16 (100%)
- **Failed:** 0

### Best Configuration

**Config:** `balanced/bfloat16`
- Latency: 0.614s ± 0.155s
- Throughput: **8.14 images/sec**

### Slowest Configuration

**Config:** `minimal/False`
- Latency: 0.993s
- Throughput: 5.03 images/sec
- **Slowdown vs best:** 1.62x

### Performance by Mixed Precision

| Precision | Configs | Avg Throughput | Avg Latency |
|-----------|---------|----------------|-------------|
| bfloat16 | 4 | 8.10 img/s | 0.617s |
| float16 | 4 | 8.05 img/s | 0.621s |
| None | 4 | 8.04 img/s | 0.622s |
| False | 4 | 5.12 img/s | 0.977s |

### Performance by Optimization Mode

| Mode | Configs | Avg Throughput | Avg Latency |
|------|---------|----------------|-------------|
| balanced | 4 | 7.37 img/s | 0.705s |
| max | 8 | 7.34 img/s | 0.707s |
| minimal | 4 | 7.25 img/s | 0.718s |

### All Configurations

| Mode | Precision | Compile | Latency (s) | Throughput (img/s) | Stability |
|------|-----------|---------|-------------|-------------------|-----------|
| balanced | bfloat16 | ✗ | 0.614 ± 0.003 | 8.14 | ✓ |
| balanced | None | ✗ | 0.615 ± 0.004 | 8.13 | ✓ |
| max | None | ✗ | 0.616 ± 0.003 | 8.11 | ✓ |
| max | bfloat16 | ✗ | 0.617 ± 0.004 | 8.10 | ✓ |
| max | bfloat16 | ✓ | 0.618 ± 0.004 | 8.09 | ✓ |
| minimal | bfloat16 | ✗ | 0.618 ± 0.006 | 8.08 | ✓ |
| max | None | ✓ | 0.620 ± 0.002 | 8.06 | ✓ |
| balanced | float16 | ✗ | 0.620 ± 0.002 | 8.06 | ✓ |
| max | float16 | ✗ | 0.620 ± 0.004 | 8.06 | ✓ |
| max | float16 | ✓ | 0.623 ± 0.003 | 8.03 | ✓ |
| minimal | float16 | ✗ | 0.623 ± 0.001 | 8.03 | ✓ |
| minimal | None | ✗ | 0.636 ± 0.006 | 7.87 | ✓ |
| max | False | ✗ | 0.970 ± 0.009 | 5.15 | ✓ |
| max | False | ✓ | 0.972 ± 0.006 | 5.15 | ✓ |
| balanced | False | ✗ | 0.972 ± 0.008 | 5.14 | ✓ |
| minimal | False | ✗ | 0.993 ± 0.006 | 5.03 | ✓ |

---

## MPS Performance Analysis

### Overview

- **Total Configurations:** 12
- **Successful:** 12/12 (100%)
- **Failed:** 0

### Best Configuration

**Config:** `minimal/bfloat16`
- Latency: 2.939s ± 0.828s
- Throughput: **1.70 images/sec**

### Slowest Configuration

**Config:** `max/None`
- Latency: 5.332s
- Throughput: 0.94 images/sec
- **Slowdown vs best:** 1.81x

### Performance by Mixed Precision

| Precision | Configs | Avg Throughput | Avg Latency |
|-----------|---------|----------------|-------------|
| bfloat16 | 4 | 1.59 img/s | 3.173s |
| False | 4 | 1.04 img/s | 4.814s |
| None | 4 | 1.04 img/s | 4.832s |

### Performance by Optimization Mode

| Mode | Configs | Avg Throughput | Avg Latency |
|------|---------|----------------|-------------|
| max | 6 | 1.24 img/s | 4.217s |
| minimal | 3 | 1.23 img/s | 4.357s |
| balanced | 3 | 1.18 img/s | 4.300s |

### All Configurations

| Mode | Precision | Compile | Latency (s) | Throughput (img/s) | Stability |
|------|-----------|---------|-------------|-------------------|-----------|
| minimal | bfloat16 | ✗ | 2.939 ± 0.079 | 1.70 | ✓ |
| max | bfloat16 | ✗ | 2.961 ± 0.266 | 1.69 | ✓ |
| max | bfloat16 | ✓ | 3.141 ± 0.250 | 1.59 | ✓ |
| balanced | bfloat16 | ✗ | 3.651 ± 0.043 | 1.37 | ✓ |
| max | None | ✗ | 4.520 ± 0.072 | 1.11 | ✓ |
| balanced | None | ✗ | 4.534 ± 0.191 | 1.10 | ✓ |
| max | False | ✗ | 4.671 ± 0.183 | 1.07 | ✓ |
| max | False | ✓ | 4.680 ± 0.124 | 1.07 | ✓ |
| balanced | False | ✗ | 4.714 ± 0.271 | 1.06 | ✓ |
| minimal | None | ✗ | 4.943 ± 0.191 | 1.01 | ✓ |
| minimal | False | ✗ | 5.190 ± 1.173 | 0.96 | ✓ |
| max | None | ✓ | 5.332 ± 0.829 | 0.94 | ✓ |

---

## CPU Performance Analysis

### Overview

- **Total Configurations:** 8
- **Successful:** 8/8 (100%)
- **Failed:** 0

### Best Configuration

**Config:** `balanced/False`
- Latency: 9.307s ± 0.447s
- Throughput: **0.54 images/sec**

### Slowest Configuration

**Config:** `balanced/None`
- Latency: 10.796s
- Throughput: 0.46 images/sec
- **Slowdown vs best:** 1.16x

### Performance by Mixed Precision

| Precision | Configs | Avg Throughput | Avg Latency |
|-----------|---------|----------------|-------------|
| False | 4 | 0.52 img/s | 9.557s |
| None | 4 | 0.50 img/s | 10.066s |

### Performance by Optimization Mode

| Mode | Configs | Avg Throughput | Avg Latency |
|------|---------|----------------|-------------|
| max | 4 | 0.52 img/s | 9.619s |
| minimal | 2 | 0.50 img/s | 9.958s |
| balanced | 2 | 0.50 img/s | 10.051s |

### All Configurations

| Mode | Precision | Compile | Latency (s) | Throughput (img/s) | Stability |
|------|-----------|---------|-------------|-------------------|-----------|
| balanced | False | ✗ | 9.307 ± 1.416 | 0.54 | ✓ |
| max | None | ✓ | 9.442 ± 1.301 | 0.53 | ✓ |
| max | False | ✗ | 9.531 ± 0.560 | 0.52 | ✓ |
| max | False | ✓ | 9.627 ± 1.362 | 0.52 | ✓ |
| minimal | False | ✗ | 9.763 ± 1.183 | 0.51 | ✓ |
| max | None | ✗ | 9.875 ± 0.991 | 0.51 | ✓ |
| minimal | None | ✗ | 10.153 ± 0.766 | 0.49 | ✓ |
| balanced | None | ✗ | 10.796 ± 0.949 | 0.46 | ✓ |

---

## Cross-Device Comparison

### Same Configuration Across Devices

Comparing `minimal/bfloat16` configuration:

| Device | Latency (s) | Throughput (img/s) | Relative Speed |
|--------|-------------|-------------------|----------------|
| CUDA | 0.618 | 8.08 | 1.00x |
| MPS | 2.939 | 1.70 | 0.21x |

*Baseline: CUDA*

## Recommendations

### Production Use Cases

#### High-Throughput Server (CUDA)
- **Configuration:** `balanced/bfloat16`
- **Expected Performance:** 8.14 img/s
- **Use Case:** Batch processing, cloud deployment

#### Mac Development/Testing (MPS)
- **Configuration:** `minimal/bfloat16`
- **Expected Performance:** 1.70 img/s
- **Use Case:** Local development, macOS applications
- **Memory Setting:** `torch.mps.set_per_process_memory_fraction(0.85)`

#### CPU Fallback
- **Configuration:** `balanced/False`
- **Expected Performance:** 0.54 img/s
- **Use Case:** No GPU available, edge devices

## Technical Notes

### torch.compile Status

**Status:** Disabled on all backends

**Reason:** Depth Anything 3 architecture is too complex for torch.compile/Triton:
- Nested models with dynamic shapes
- Complex geometric operations (pose estimation, multi-view)
- Deep expression trees that Triton cannot optimize

**Error (CUDA/MPS):**
```
triton.compiler.errors.CompilationError: Complex expression tree
Metal Error: Threadgroup memory size exceeds maximum
```

### Optimization Status

✅ **Implemented:**
- Device-specific optimizers (CPU/MPS/CUDA)
- Mixed precision inference (FP16/BF16)
- Adaptive batch sizing
- Memory management (MPS: 0.85 fraction)
- CPU threading optimization
- CUDA: cuDNN benchmark, TF32

⏳ **Potential Future Optimizations:**
- Model quantization (INT8)
- ONNX Runtime export
- TensorRT (CUDA)
- CoreML + ANE (Apple Silicon)
- Model caching

## Methodology

### Test Environment

**Hardware:**
- CPU: Multi-core processor
- GPU (CUDA): NVIDIA GPU
- GPU (MPS): Apple Silicon (Unified Memory Architecture)

**Software:**
- PyTorch 2.x
- CUDA Toolkit (for CUDA tests)
- macOS (for MPS tests)

### Benchmark Protocol

1. **Warmup:** 2 iterations to initialize kernels and caches
2. **Measurement:** 3-5 timed iterations
3. **Input:** 5 images per iteration (504x504 resolution)
4. **Metrics:**
   - Latency: Mean ± Std Dev (seconds)
   - Throughput: Images per second
   - Numerical stability: Max depth difference across runs

### Configuration Matrix

**Performance Modes:**
- `minimal`: Basic optimizations
- `balanced`: Moderate optimizations
- `max`: Aggressive optimizations

**Mixed Precision:**
- `False` / `FP32`: Full precision (baseline)
- `None`: Device default
- `float16`: Half precision
- `bfloat16`: Brain float (better stability)

**torch.compile:**
- Tested but disabled due to incompatibility

---

*Report generated automatically from benchmark results*
