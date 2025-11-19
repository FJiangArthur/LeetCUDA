# Project: Optimized Matrix Multiplication Library

## Overview

**Level**: L2-L3 (Intermediate to Advanced)
**Estimated Time**: 15-25 hours
**Difficulty**: ⭐⭐⭐ (Challenging)
**Type**: Performance Optimization

Build a progressively optimized matrix multiplication library, implementing techniques from naive to near-cuBLAS performance.

---

## Learning Objectives

By completing this project, you will:
- ✅ Implement multiple GEMM optimization levels
- ✅ Master shared memory tiling strategies
- ✅ Apply register blocking for performance
- ✅ Achieve 50-80% of cuBLAS performance
- ✅ Benchmark and profile CUDA kernels
- ✅ Understand the full optimization pipeline

---

## Prerequisites

**Required Knowledge**:
- L2: Shared memory and tiling (See: [tutorials/intermediate/shared-memory-tiling.md](../../../tutorials/intermediate/shared-memory-tiling.md))
- L2: Memory coalescing
- L1: Thread hierarchy and indexing
- Linear algebra basics (matrix multiplication)

**Required Skills**:
- Write intermediate CUDA kernels
- Use NSight Systems/Compute for profiling
- Analyze performance bottlenecks

**Software Requirements**:
- CUDA >= 11.0
- PyTorch >= 2.0
- Python >= 3.8
- NSight Systems (for profiling)

---

## Problem Statement

### Background

Matrix multiplication (GEMM: General Matrix-Matrix Multiply) is the **most critical operation** in deep learning:
- 90%+ of training time
- Core of convolutions, attention, fully-connected layers
- Optimization directly impacts training speed

**Challenge**: cuBLAS achieves ~30 TFLOPS (FP32) on A100. Can you reach even 50% of that?

### The Task

Implement **5 progressively optimized versions** of matrix multiplication:

1. **Naive** - Baseline (slow but correct)
2. **Tiled** - Use shared memory (10-20x faster)
3. **Register Blocked** - Compute multiple outputs per thread (50-100x faster)
4. **Vectorized** - Use float4 loads (2x faster than previous)
5. **Polished** - Add all optimizations (aim for 50-80% of cuBLAS)

---

## Functional Requirements

### Must Have (Core Functionality)

#### 1. Matrix Multiplication Kernel Versions

Implement these 5 versions with increasing complexity:

**Version 0: Naive**
```cuda
C[i][j] = sum(A[i][k] * B[k][j] for k in range(K))
// Each thread computes one output element
// All global memory accesses (slow!)
```

**Version 1: Shared Memory Tiled (16×16 or 32×32)**
```cuda
- Load tiles of A and B into shared memory
- Compute using shared memory (100x faster reads)
- Each thread still computes one output
```

**Version 2: Register Blocking (Thread Tile 4×4 or 8×8)**
```cuda
- Each thread computes TM×TN outputs (e.g., 8×8 = 64 outputs)
- Accumulate in registers (fastest memory!)
- Reduces shared memory traffic
```

**Version 3: Vectorized Loads**
```cuda
- Load 4 floats per instruction (float4)
- Fewer memory instructions
- Better bandwidth utilization
```

**Version 4: Fully Optimized**
```cuda
- Combine all techniques
- Add bank conflict avoidance (padding)
- Loop unrolling
- Optimal block sizes
```

#### 2. PyTorch Integration

```python
import torch
from gemm_lib import gemm_naive, gemm_tiled, gemm_blocked, gemm_optimized

A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')

# Test each version
C_naive = gemm_naive(A, B)
C_tiled = gemm_tiled(A, B)
C_blocked = gemm_blocked(A, B)
C_optimized = gemm_optimized(A, B)

# All should match cuBLAS (within tolerance)
C_cublas = A @ B
```

#### 3. Testing Framework

- **Correctness**: Compare with cuBLAS (tolerance: 1e-3 for FP32)
- **Multiple sizes**: Powers of 2 and odd sizes
- **Rectangular matrices**: M≠N≠K
- **Edge cases**: Very small, very large matrices

#### 4. Performance Benchmarking

Measure and compare:
- **Time** (milliseconds)
- **GFLOPS** (Giga-FLOPs per second)
- **Bandwidth** (GB/s)
- **Efficiency** (% of cuBLAS)

---

## Technical Specifications

### Project Structure

```
matrix-multiply-library/
├── src/
│   ├── gemm_naive.cu          # Version 0: Baseline
│   ├── gemm_tiled.cu          # Version 1: Shared memory
│   ├── gemm_blocked.cu        # Version 2: Register blocking
│   ├── gemm_vectorized.cu     # Version 3: Float4 loads
│   ├── gemm_optimized.cu      # Version 4: All optimizations
│   └── common.h               # Common utilities
│
├── python/
│   ├── __init__.py
│   ├── gemm.py                # PyTorch bindings
│   └── launch_config.py       # Grid/block configuration
│
├── tests/
│   ├── test_correctness.py    # Verify vs cuBLAS
│   ├── test_shapes.py         # Different matrix shapes
│   └── test_edge_cases.py     # Boundary conditions
│
├── benchmarks/
│   ├── benchmark_all.py       # Compare all versions
│   ├── benchmark_scaling.py   # Test different sizes
│   └── profile.py             # NSight integration
│
├── docs/
│   ├── optimization_guide.md  # Your learnings
│   └── performance_report.md  # Results and analysis
│
├── setup.py
└── README.md
```

---

## Implementation Roadmap

### Phase 1: Naive Implementation (2-3 hours)

**Goal**: Working baseline, establish testing framework

**Tasks**:
1. [ ] Implement naive GEMM kernel
   ```cuda
   C[row][col] = sum(A[row][k] * B[k][col])
   ```
2. [ ] Create PyTorch binding
3. [ ] Write correctness test vs cuBLAS
4. [ ] Benchmark baseline performance

**Validation**:
- [ ] Correctness: Max error < 1e-3
- [ ] Performance: ~50-100 GFLOPS (1-5% of cuBLAS)

**Expected Performance** (1024×1024, A100):
- Time: ~20 ms
- GFLOPS: ~100 (vs cuBLAS's ~30,000)

---

### Phase 2: Shared Memory Tiling (4-6 hours)

**Goal**: 10-20x speedup through shared memory

**Tasks**:
1. [ ] Implement tiled version with 16×16 tiles
   ```cuda
   - Load tiles of A and B to shared memory
   - Sync threads
   - Compute using shared memory
   - Sync before next tile
   ```
2. [ ] Handle boundary conditions (matrices not divisible by tile size)
3. [ ] Test correctness
4. [ ] Benchmark and compare to naive

**Key Concepts**:
```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// Each tile reduces global memory accesses by TILE_SIZE
// Speedup ≈ TILE_SIZE (if memory-bound)
```

**Validation**:
- [ ] Correctness maintained
- [ ] Performance: ~1,000-2,000 GFLOPS (10-20x improvement)

**Checkpoints**:
- Can you explain why shared memory is faster?
- What happens if you remove `__syncthreads()`?
- Try different tile sizes (8, 16, 32) - which is best?

---

### Phase 3: Register Blocking (5-8 hours)

**Goal**: 50-100x speedup total through register accumulation

**Tasks**:
1. [ ] Implement register-blocked version
   - Each thread computes TM×TN outputs (e.g., 8×8)
   - Accumulate in register variables
2. [ ] Optimize inner loop structure
3. [ ] Handle non-divisible dimensions
4. [ ] Profile with NSight Compute

**Algorithm**:
```cuda
// Each thread computes 8×8 outputs
float accum[8][8] = {0.0f};

for each tile:
    load_tile_to_shared();
    __syncthreads();

    for k in tile:
        // Load 8 elements from A, 8 from B
        float a_reg[8], b_reg[8];
        for i in 8:
            a_reg[i] = As[ty*8+i][k];
            b_reg[i] = Bs[k][tx*8+i];

        // Outer product: accumulate to registers
        for i in 8:
            for j in 8:
                accum[i][j] += a_reg[i] * b_reg[j];

    __syncthreads();

// Write accumulated results
for i,j in 8×8:
    C[row+i][col+j] = accum[i][j];
```

**Validation**:
- [ ] Correctness maintained
- [ ] Performance: ~5,000-10,000 GFLOPS (50-100x vs naive!)

**Deep Dive Questions**:
- Why do registers help more than shared memory?
- What's the optimal TM×TN? (Try 4×4, 8×8, 16×16)
- How does this affect occupancy?

---

### Phase 4: Vectorized Loads (2-3 hours)

**Goal**: 1.5-2x speedup through better memory instructions

**Tasks**:
1. [ ] Use float4 for loading (4 floats per instruction)
   ```cuda
   float4 a_vec = reinterpret_cast<float4*>(&A[index])[0];
   ```
2. [ ] Ensure alignment (pointers must be 16-byte aligned)
3. [ ] Update tile loading code
4. [ ] Measure improvement

**Key Code**:
```cuda
// Instead of 4 loads:
As[ty][tx+0] = A[...];
As[ty][tx+1] = A[...];
As[ty][tx+2] = A[...];
As[ty][tx+3] = A[...];

// One vectorized load:
float4 vec = *reinterpret_cast<float4*>(&A[...]);
As[ty][tx+0] = vec.x;
As[ty][tx+1] = vec.y;
As[ty][tx+2] = vec.z;
As[ty][tx+3] = vec.w;
```

**Validation**:
- [ ] Performance: ~8,000-15,000 GFLOPS

---

### Phase 5: Final Optimizations (3-5 hours)

**Goal**: Reach 50-80% of cuBLAS performance

**Tasks**:
1. [ ] Add padding to avoid bank conflicts
   ```cuda
   __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 for padding
   ```
2. [ ] Unroll inner loops (`#pragma unroll`)
3. [ ] Tune block sizes (try 64×64, 128×128)
4. [ ] Profile and optimize hotspots
5. [ ] Compare with cuBLAS across sizes

**Target Performance** (1024×1024, FP32):
- A100: 15,000-25,000 GFLOPS (50-80% of cuBLAS)
- RTX 3080: 10,000-15,000 GFLOPS (50-70% of cuBLAS)

---

## Performance Targets

### Minimum Requirements (Pass)

| Version | GFLOPS (1024²) | vs cuBLAS | Speedup vs Naive |
|---------|----------------|-----------|------------------|
| Naive | 50-100 | 1% | 1x |
| Tiled | 500+ | 5%+ | 10x+ |
| Blocked | 3,000+ | 20%+ | 60x+ |
| Vectorized | 5,000+ | 30%+ | 100x+ |
| Optimized | 10,000+ | 50%+ | 200x+ |

### Stretch Goals (Excellent)

| Version | GFLOPS (1024²) | vs cuBLAS |
|---------|----------------|-----------|
| Optimized | 20,000+ | 70%+ |
| With FP16 (bonus) | 100,000+ | 80%+ |

---

## Benchmarking Framework

### Example Benchmark Code

```python
import torch
import time
from gemm_lib import *

def benchmark_gemm(gemm_func, name, sizes, iterations=100):
    """Benchmark GEMM function across different sizes."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    print(f"{'Size':<15} {'Time (ms)':<12} {'GFLOPS':<12} {'vs cuBLAS':<12}")
    print("-" * 60)

    results = []

    for M, N, K in sizes:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(10):
            C = gemm_func(A, B)
        torch.cuda.synchronize()

        # Benchmark custom
        start = time.time()
        for _ in range(iterations):
            C = gemm_func(A, B)
        torch.cuda.synchronize()
        time_custom = (time.time() - start) / iterations * 1000  # ms

        # Benchmark cuBLAS
        start = time.time()
        for _ in range(iterations):
            C_ref = A @ B
        torch.cuda.synchronize()
        time_cublas = (time.time() - start) / iterations * 1000  # ms

        # Calculate metrics
        flops = 2 * M * N * K
        gflops_custom = flops / (time_custom * 1e6)
        gflops_cublas = flops / (time_cublas * 1e6)
        efficiency = 100 * gflops_custom / gflops_cublas

        print(f"{M}×{N}×{K:<6} {time_custom:<12.3f} {gflops_custom:<12.1f} {efficiency:<12.1f}%")

        results.append({
            'size': (M, N, K),
            'time': time_custom,
            'gflops': gflops_custom,
            'efficiency': efficiency
        })

    return results


if __name__ == "__main__":
    # Test sizes
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    # Benchmark all versions
    results_naive = benchmark_gemm(gemm_naive, "Naive", sizes)
    results_tiled = benchmark_gemm(gemm_tiled, "Tiled", sizes)
    results_blocked = benchmark_gemm(gemm_blocked, "Register Blocked", sizes)
    results_optimized = benchmark_gemm(gemm_optimized, "Optimized", sizes)

    # Summary plot
    plot_results([results_naive, results_tiled, results_blocked, results_optimized])
```

---

## Grading Rubric

### Excellent (90-100%)
- ✅ All 5 versions implemented and working
- ✅ All tests pass (100% correctness)
- ✅ Optimized version reaches 60%+ of cuBLAS
- ✅ Comprehensive benchmarks and analysis
- ✅ Clean, well-documented code
- ✅ Insightful performance report

### Good (75-89%)
- ✅ 4 versions working (naive through vectorized)
- ✅ Correctness tests pass
- ✅ Optimized version reaches 40%+ of cuBLAS
- ✅ Basic benchmarks provided
- ✅ Code is functional and documented

### Satisfactory (60-74%)
- ✅ 3 versions working (naive, tiled, blocked)
- ✅ Most tests pass
- ✅ Some optimization achieved (20%+ of cuBLAS)
- ✅ Basic documentation

### Needs Improvement (<60%)
- ❌ Fewer than 3 versions working
- ❌ Correctness issues
- ❌ Performance < 10% of cuBLAS

---

## Extensions & Challenges

### Extension 1: Support Different Data Types (Medium)

Implement FP16, BF16, INT8 versions:
```python
C_fp16 = gemm_optimized(A.half(), B.half())
C_bf16 = gemm_optimized(A.bfloat16(), B.bfloat16())
```

**Challenge**: Achieve 2-3x speedup with FP16!

### Extension 2: Batched GEMM (Medium)

Process multiple small matrices simultaneously:
```python
# Batch of 100 small matrices
C_batch = gemm_batched(A_batch, B_batch)  # [100, 128, 128]
```

### Extension 3: Tensor Cores (Hard)

Use WMMA for FP16:
```cuda
#include <mma.h>
using namespace nvcuda::wmma;
// Implement using 16×16×16 matrix fragments
```

**Challenge**: Reach 80%+ of cuBLAS with Tensor Cores!

### Extension 4: Auto-Tuning (Hard)

Automatically find best tile sizes:
```python
best_config = auto_tune(matrix_shape=(4096, 4096, 4096))
# Tests different block sizes, tile sizes, etc.
```

### Challenge: Beat cuBLAS

Can you beat cuBLAS for any specific matrix size or shape?
(Hint: cuBLAS is general-purpose; you can specialize!)

---

## Common Pitfalls

### Pitfall 1: Bank Conflicts

**Symptom**: Shared memory version slower than expected

**Solution**:
```cuda
// Add padding
__shared__ float As[TILE][TILE + 1];  // +1 avoids conflicts
```

### Pitfall 2: Incorrect Boundary Handling

**Symptom**: Wrong results for non-power-of-2 sizes

**Solution**: Always check bounds!
```cuda
if (row < M && col < N && k < K) {
    // Safe to access
}
```

### Pitfall 3: Register Spilling

**Symptom**: Register-blocked version slower than expected

**Check**:
```bash
nvcc --ptxas-options=-v kernel.cu
# Look for "registers spilled to local memory"
```

**Solution**: Reduce TM×TN or use #pragma unroll sparingly

---

## Profiling Guide

### Using NSight Compute

```bash
# Profile kernel
ncu --set full -o profile python benchmark.py

# Key metrics to check:
# 1. SM Efficiency (aim for >80%)
# 2. Memory Throughput (aim for >80% of peak)
# 3. Warp Execution Efficiency
# 4. Register usage
# 5. Shared memory bank conflicts
```

### Using NSight Systems

```bash
# Timeline view
nsys profile -o timeline python benchmark.py

# Check:
# 1. Kernel launch overhead
# 2. Memory transfer times
# 3. Kernel execution time
```

---

## Resources

### Papers & Guides
- [GEMM Optimization Paper](../../papers/gemm-optimization/matrix-multiplication-optimization.md)
- [Shared Memory Tutorial](../../tutorials/intermediate/shared-memory-tiling.md)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)

### Reference Code
- LeetCUDA: `kernels/sgemm/`, `kernels/hgemm/`
- cuBLAS: NVIDIA's optimized library
- PyTorch source: `aten/src/ATen/native/cuda/`

---

## Submission

### Required Deliverables

1. **Source code** (all versions working)
2. **Tests** (all passing)
3. **Benchmarks** (comprehensive results)
4. **Performance report** (2-3 pages, see template)

### Performance Report Template

```markdown
# GEMM Library Performance Report

## Implementation Summary
- Versions implemented: [list]
- Best performance achieved: [GFLOPS] ([X]% of cuBLAS)
- Development time: [hours]

## Optimization Journey

### Version 0: Naive
- Performance: [GFLOPS]
- Key characteristics: [describe]

### Version 1: Tiled
- Performance: [GFLOPS]
- Speedup: [X]x
- Tile size used: [value]
- Key insight: [what you learned]

[... for each version]

## Performance Analysis

### Benchmark Results
[Table showing performance across sizes]

### Bottleneck Analysis
- Memory-bound or compute-bound? [analysis]
- NSight profiling insights: [key findings]
- Remaining optimization opportunities: [list]

## Challenges & Solutions

1. [Challenge]: [How you solved it]
2. [Challenge]: [How you solved it]

## Key Learnings

- [Learning 1]
- [Learning 2]
- [Learning 3]

## Future Work

- [What would you optimize next]
```

---

**Good luck! This project will transform your understanding of GPU optimization!**

---

**Project Designed by**: Project Designer Agent #5
**Session**: session_20251119_051648
**Difficulty**: Intermediate-Advanced (L2-L3)
**Time**: 15-25 hours
**Critical Skill**: Foundation for all high-performance GPU programming
