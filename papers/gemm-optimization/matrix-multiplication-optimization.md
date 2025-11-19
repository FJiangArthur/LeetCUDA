# Paper: Optimizing Matrix Multiplication (GEMM)

## Metadata
- **Topic**: General Matrix Multiply (GEMM) Optimization
- **Based on**: Multiple research papers and NVIDIA optimization guides
- **Level**: L2-L3 (Intermediate to Advanced)
- **Difficulty**: ⭐⭐⭐
- **Implementation Status**: 💻 Available in `kernels/sgemm/`, `kernels/hgemm/`

---

## TL;DR

Matrix multiplication (GEMM) is the **most important operation** in deep learning. Optimizing GEMM from naive (~50 GFLOPS) to near-cuBLAS performance (~30,000 GFLOPS on A100) requires a systematic progression through:
1. Tiling for cache
2. Register blocking
3. Vectorized loads/stores
4. Warp specialization
5. Tensor Core utilization

---

## Why GEMM Matters

### The Fundamental Operation

**GEMM**: General Matrix-Matrix Multiply
```
C = α·(A @ B) + β·C

Where:
- A: M×K matrix
- B: K×N matrix
- C: M×N matrix (output)
- α, β: scalars
```

### Ubiquity in Deep Learning

**Every major operation uses GEMM**:

```python
# Linear layers
output = input @ weights  # GEMM!

# Convolution (via im2col)
output = im2col(input) @ filters  # GEMM!

# Attention
scores = Q @ K.T  # GEMM!
output = scores @ V  # GEMM!

# Batch normalization gradients
grad_weight = input.T @ grad_output  # GEMM!
```

**Performance Impact**:
- 90%+ of training time is GEMM
- 10x better GEMM → 10x faster training
- cuBLAS vs naive: **100-1000x speedup**!

---

## GEMM Complexity

### Computational Complexity

**Operation Count**:
```
For C = A @ B (M×N result, K inner dimension):

FLOPs = 2·M·N·K
(Multiply-add for each of M·N output elements, K operations each)

Example: 1024×1024 matrices
FLOPs = 2 · 1024³ = 2.15 billion FLOPs
```

### Memory Complexity

**Memory Required**:
```
A: M×K elements
B: K×N elements
C: M×N elements

Total: (M·K + K·N + M·N) × sizeof(element)

Example (FP32): 1024×1024
Memory = 3 · 1024² · 4 bytes = 12 MB
```

### Arithmetic Intensity

**Key Metric**: FLOPs per byte
```
Arithmetic Intensity = FLOPs / Bytes Accessed
                     = 2·M·N·K / (M·K + K·N + M·N)

For square matrices (M=N=K):
AI = 2·N³ / 3·N² = (2/3)·N

Example: N=1024
AI = 682 FLOPs/byte (very high!)
```

**Implication**: GEMM can be **compute-bound** (not memory-bound) if optimized!

---

## Optimization Journey

### Version 0: Naive (Baseline)

**Algorithm**:
```cuda
__global__ void gemmNaive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Performance**: ~50-100 GFLOPS on A100
**Issues**:
- Uncoalesced memory access for A
- Each element of A and B loaded K times and N times respectively
- No data reuse
- **Efficiency**: <1% of peak!

---

### Version 1: Tiled GEMM with Shared Memory

**Key Idea**: Load tiles into shared memory, reuse for multiple outputs

```cuda
#define TILE_SIZE 16

__global__ void gemmTiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        // Load tile of B
        if (t * TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Performance**: ~500-1000 GFLOPS (10x improvement!)
**Improvements**:
- Each tile loaded once, reused TILE_SIZE times
- Coalesced global memory access
- Shared memory is 100x faster

**Remaining Issues**:
- Still reading shared memory frequently
- Not utilizing registers fully

---

### Version 2: Register Tiling (Blocking)

**Key Idea**: Each thread computes multiple output elements, accumulating in registers

```cuda
#define BM 64  // Block size M
#define BN 64  // Block size N
#define BK 8   // Block size K
#define TM 8   // Thread tile M
#define TN 8   // Thread tile N

__global__ void gemmRegisterTiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Each thread computes TM×TN output elements
    float accum[TM][TN] = {0.0f};

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty * TM;
    int col = blockIdx.x * BN + tx * TN;

    // Loop over K dimension in chunks
    for (int k = 0; k < K; k += BK) {
        // Cooperatively load tiles into shared memory
        // ... (loading code)

        __syncthreads();

        // Compute TM×TN outputs using shared memory
        for (int ki = 0; ki < BK; ki++) {
            // Load column of A (TM elements)
            float a[TM];
            for (int i = 0; i < TM; i++) {
                a[i] = As[ty * TM + i][ki];
            }

            // Load row of B (TN elements)
            float b[TN];
            for (int j = 0; j < TN; j++) {
                b[j] = Bs[ki][tx * TN + j];
            }

            // Outer product: accumulate to registers
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    accum[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + (col + j)] = accum[i][j];
            }
        }
    }
}
```

**Performance**: ~3,000-5,000 GFLOPS (50-100x vs naive!)
**Improvements**:
- Accumulation in registers (fastest memory!)
- Better instruction-level parallelism
- Reduced shared memory traffic

**Key Insight**:
```
Naive:       1 thread = 1 output
Tiled:       1 thread = 1 output (but uses shared mem)
Reg-blocked: 1 thread = TM×TN outputs (64 for TM=TN=8!)
```

---

### Version 3: Vectorized Memory Access

**Key Idea**: Load 128 bits (4×FP32) per instruction

```cuda
// Load 4 floats at once
float4 a_vec = reinterpret_cast<float4*>(&A[index])[0];

// Equivalent to 4 separate loads, but 1 instruction!
float a0 = a_vec.x;
float a1 = a_vec.y;
float a2 = a_vec.z;
float a3 = a_vec.w;
```

**Performance**: ~6,000-8,000 GFLOPS
**Improvement**: Fewer memory instructions, better bandwidth utilization

---

### Version 4: Warp-Level Optimizations

**Techniques**:
1. **Warp shuffles** for data exchange (no shared memory)
2. **Warp specialization**: Different warps do different tasks
3. **Careful instruction scheduling** to hide latency

**Performance**: ~10,000-15,000 GFLOPS (50% of peak FP32)

---

### Version 5: Tensor Cores (FP16/BF16)

**Ultimate Optimization**: Use specialized hardware

```cuda
// WMMA API (Warp Matrix Multiply-Accumulate)
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void gemmWMMA(half* A, half* B, float* C, int M, int N, int K) {
    // Declare fragments (registers)
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += 16) {
        // Load matrix fragments
        load_matrix_sync(a_frag, A + ..., K);
        load_matrix_sync(b_frag, B + ..., N);

        // Perform matrix multiply-accumulate
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(C + ..., c_frag, N, mem_row_major);
}
```

**Performance**: ~300,000+ GFLOPS on A100 (FP16 with Tensor Cores!)
**Speedup**: 3000x vs naive, 100x vs basic tiling!

---

## Performance Progression Summary

| Version | Technique | A100 Performance | vs Naive | % of Peak |
|---------|-----------|------------------|----------|-----------|
| V0 | Naive | 50 GFLOPS | 1x | 0.3% |
| V1 | Shared Memory Tiling | 1,000 GFLOPS | 20x | 5% |
| V2 | Register Blocking | 5,000 GFLOPS | 100x | 25% |
| V3 | Vectorized Loads | 8,000 GFLOPS | 160x | 40% |
| V4 | Warp Optimizations | 15,000 GFLOPS | 300x | 75% |
| V5 | Tensor Cores (FP16) | 300,000 GFLOPS | 6000x | 96% |
| cuBLAS | Production Library | 312,000 GFLOPS | 6240x | 100% |

**Peak A100 FP16 (Tensor Cores)**: ~312 TFLOPS

---

## Key Optimizations Explained

### 1. Tiling for Cache Hierarchy

**Principle**: Divide work to fit in faster memory levels

```
Level 1: Tile for L2 cache (MB-scale)
  └─ Level 2: Tile for shared memory (KB-scale)
      └─ Level 3: Tile for registers (bytes-scale)
```

**Example Dimensions**:
```
Global: 4096×4096 (67 MB for FP32)
  └─ L2 tile: 128×128 (256 KB)
      └─ Shared memory tile: 64×64 (16 KB)
          └─ Register tile: 8×8 (256 bytes)
```

### 2. Memory Access Patterns

**Coalescing**:
```cuda
// BAD: Strided access
for (int i = 0; i < N; i++) {
    float val = A[i * stride];  // Uncoalesced!
}

// GOOD: Sequential access
for (int i = 0; i < N; i++) {
    float val = A[i];  // Coalesced!
}
```

**Padding for Bank Conflict Avoidance**:
```cuda
// Without padding: Bank conflicts
__shared__ float tile[16][16];

// With padding: Conflict-free
__shared__ float tile[16][17];  // Extra column
```

### 3. Instruction-Level Parallelism

**Unroll inner loops**:
```cuda
// Before: Loop overhead
for (int k = 0; k < 8; k++) {
    sum += a[k] * b[k];
}

// After: Unrolled (compiler does this with #pragma unroll)
sum += a[0] * b[0];
sum += a[1] * b[1];
sum += a[2] * b[2];
// ... all 8 explicitly
```

**Benefit**: More instruction-level parallelism, hides latency

---

## Roofline Model for GEMM

### What is Roofline Model?

**Visualization** of performance limits:

```
Performance (GFLOPS)
     ▲
     │           ──────────────────────── Compute Bound (Peak FLOPS)
     │         ╱
     │       ╱
     │     ╱  Memory Bound
     │   ╱    (Bandwidth × Arithmetic Intensity)
     │ ╱
     └────────────────────────────────────> Arithmetic Intensity (FLOPS/byte)
```

### For GEMM on A100

**Parameters**:
- Peak FP32: 19.5 TFLOPS
- Memory Bandwidth: 1,555 GB/s
- Arithmetic Intensity: (2/3)·N FLOPS/byte (for N×N)

**Analysis**:
```
For N=1024:
AI = 682 FLOPS/byte

Memory-bound limit: 1,555 GB/s × 682 = 1,060,510 GFLOPS
Compute-bound limit: 19,500 GFLOPS

Actual limit: min(1,060,510, 19,500) = 19,500 GFLOPS (compute-bound!)
```

**Conclusion**: For large matrices, GEMM is **compute-bound**, not memory-bound!

This is why Tensor Cores (boosting compute) help so much.

---

## Practical Implementation Guide

### Step-by-Step Optimization

**Phase 1: Get it Working** (1-2 hours)
1. Implement naive version
2. Verify correctness vs. cuBLAS
3. Benchmark baseline performance

**Phase 2: Shared Memory** (2-4 hours)
1. Implement tiled version (16×16 or 32×32 tiles)
2. Add boundary handling
3. Target: 10-20x speedup

**Phase 3: Register Blocking** (4-8 hours)
1. Each thread computes 4×4 or 8×8 outputs
2. Accumulate in registers
3. Target: 50-100x speedup

**Phase 4: Polish** (4-8 hours)
1. Vectorized loads (float4)
2. Loop unrolling
3. Padding for bank conflicts
4. Target: 100-200x speedup (50-70% of cuBLAS)

**Phase 5: Tensor Cores** (8-16 hours)
1. Switch to FP16/BF16
2. Use WMMA or MMA API
3. Target: 90-99% of cuBLAS

**Total**: 20-40 hours to reach near-cuBLAS performance!

---

## Code Example: Complete Tiled GEMM

See `kernels/sgemm/sgemm_tiled.cu` for complete implementation

**Usage**:
```python
import torch
from sgemm import gemm_tiled

A = torch.randn(2048, 2048, device='cuda')
B = torch.randn(2048, 2048, device='cuda')

# Custom GEMM
C_custom = gemm_tiled(A, B)

# cuBLAS reference
C_cublas = A @ B

# Verify
print(f"Max error: {(C_custom - C_cublas).abs().max().item()}")

# Benchmark
import time
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    C = gemm_tiled(A, B)
torch.cuda.synchronize()
elapsed = (time.time() - start) / 100

# Calculate GFLOPS
flops = 2 * 2048**3
gflops = flops / (elapsed * 1e9)
print(f"Performance: {gflops:.2f} GFLOPS")

# Compare with cuBLAS
start = time.time()
for _ in range(100):
    C = A @ B
torch.cuda.synchronize()
elapsed_cublas = (time.time() - start) / 100
gflops_cublas = flops / (elapsed_cublas * 1e9)

print(f"cuBLAS: {gflops_cublas:.2f} GFLOPS")
print(f"Efficiency: {100 * gflops / gflops_cublas:.1f}% of cuBLAS")
```

---

## Common Pitfalls

### 1. Bank Conflicts

**Problem**: Multiple threads access same shared memory bank
```cuda
// BAD: All threads read from same column
__shared__ float data[32][32];
float val = data[threadIdx.x][0];  // Bank conflict!

// GOOD: Threads read from same row
float val = data[0][threadIdx.x];  // Coalesced!
```

**Solution**: Padding or transpose in shared memory

### 2. Incorrect Boundary Handling

**Problem**: Threads exceed matrix boundaries
```cuda
// WRONG: May write out of bounds
C[row * N + col] = sum;

// CORRECT: Check boundaries
if (row < M && col < N) {
    C[row * N + col] = sum;
}
```

### 3. Suboptimal Tile Sizes

**Problem**: Tiles don't fit well in shared memory

**Guidelines**:
- Tile size should be multiple of 16 (for Tensor Cores)
- Total shared memory per block: < 48 KB (check GPU specs)
- For FP32: 32×32 tile = 4 KB, 64×64 = 16 KB

---

## Further Optimizations

### Advanced Techniques (cuBLAS-level):

1. **Double Buffering**: Overlap compute and memory loads
2. **Warp Specialization**: Some warps load, others compute
3. **Software Pipelining**: Multi-stage pipeline
4. **Auto-tuning**: Search for optimal parameters
5. **Mixed Precision**: FP16 compute, FP32 accumulate
6. **Batch GEMM**: Process multiple small matrices together

---

## Learning Path

### Prerequisites
1. ✅ [Shared Memory](../../tutorials/intermediate/shared-memory-tiling.md)
2. ✅ [Thread Hierarchy](../../tutorials/beginner/thread-hierarchy.md)
3. ✅ [Memory Coalescing](../../tutorials/intermediate/memory-coalescing.md)

### Tutorials
1. ✅ `tutorials/advanced/gemm-naive.md` - Start here
2. ✅ `tutorials/advanced/gemm-tiled.md` - Shared memory
3. ✅ `tutorials/advanced/gemm-registers.md` - Register blocking
4. ✅ `tutorials/expert/gemm-wmma.md` - Tensor Cores

### Projects
1. ✅ [GEMM Library Project](../../projects/advanced/gemm-library/) - Complete implementation

**Estimated Time**: 40-60 hours for mastery

---

## Key Takeaways

1. **GEMM is fundamental** - Most important operation in deep learning
2. **Systematic optimization** - Each technique builds on previous
3. **Roofline analysis** - Understand if memory or compute bound
4. **Memory hierarchy** - Tile for registers → shared mem → L2 → global
5. **Tensor Cores** - 100x speedup for FP16/BF16 operations
6. **Practice required** - Understanding ≠ implementation skill

---

## References

### Papers
- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn, 2008)
- "Kernel Optimization for Deep Learning on GPU" (various NVIDIA)
- "CUTLASS: Fast Linear Algebra in CUDA C++" (NVIDIA)

### Resources
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [Matrix Multiplication Background](https://siboehm.com/articles/22/CUDA-MMM)

### Implementations
- LeetCUDA: `kernels/sgemm/`, `kernels/hgemm/`
- CUTLASS: Production-quality templates
- cuBLAS: NVIDIA's optimized library

---

**Paper Summary by**: Content Creator Agent #3
**Session**: session_20251119_051648
**Level**: L2-L3
**Time to Master**: 40-60 hours
**Related Project**: [GEMM Library](../../projects/advanced/gemm-library/)
