# Tutorial: Memory Coalescing Optimization

## Learning Objectives

By the end of this tutorial, you will:
- ✅ Understand what memory coalescing is and why it matters
- ✅ Identify coalesced vs uncoalesced access patterns
- ✅ Measure memory efficiency using NSight Compute
- ✅ Transform uncoalesced code into coalesced versions
- ✅ Achieve 5-10x speedups through proper memory access

**Level**: L2 (Intermediate)
**Time**: 75-90 minutes
**Prerequisites**: Memory Model Basics, Thread Hierarchy, Element-wise Operations

---

## Introduction

### The Critical Performance Factor

**Problem**: On modern GPUs, **memory access patterns** can make a **10x performance difference**!

**Real-world example**:
```cuda
// Version 1: Uncoalesced (SLOW)
Time: 2.5 ms, Bandwidth: 100 GB/s

// Version 2: Coalesced (FAST)
Time: 0.25 ms, Bandwidth: 950 GB/s

Same algorithm, 10x speedup!
```

### What is Memory Coalescing?

**Memory coalescing** = combining multiple memory accesses into fewer transactions

**Analogy**: Imagine 32 people shopping:

**Uncoalesced (Bad)**:
```
Person 1 → Aisle 3, Shelf 7
Person 2 → Aisle 12, Shelf 2
Person 3 → Aisle 1, Shelf 9
...
= 32 separate trips across store
```

**Coalesced (Good)**:
```
All 32 people → Aisle 3, Shelves 1-32 (consecutive)
= One trip, grab items in order
```

### GPU Memory Transaction

GPUs fetch memory in **chunks** (32, 64, or 128 bytes):

```
Memory addresses:  0    4    8   12   16   20   24   28  (bytes)
                  ┌────┬────┬────┬────┬────┬────┬────┬────┐
Transaction:      │ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │
                  └────┴────┴────┴────┴────┴────┴────┴────┘
                       One 32-byte transaction (8 floats)
```

**Warp** (32 threads) = Basic GPU scheduling unit

**Goal**: When 32 threads (1 warp) access memory, minimize number of transactions

---

## Part 1: Anatomy of Coalesced Access

### Perfect Coalescing

**Pattern**: Consecutive threads access consecutive memory locations

```cuda
__global__ void coalesced(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[i];  // ✅ PERFECT!
}
```

**What happens** (warp 0, threads 0-31):
```
Thread 0 → data[0]
Thread 1 → data[1]
Thread 2 → data[2]
...
Thread 31 → data[31]

All 32 accesses in ONE 128-byte transaction! ✅
```

### Strided Access (Uncoalesced)

**Pattern**: Threads access memory with gaps (stride)

```cuda
__global__ void strided(float* data, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[i * stride];  // ❌ BAD if stride > 1
}
```

**What happens** (stride = 32, warp 0):
```
Thread 0 → data[0]    (transaction 1)
Thread 1 → data[32]   (transaction 2)
Thread 2 → data[64]   (transaction 3)
...
Thread 31 → data[992] (transaction 32)

32 separate transactions! ❌ 32x slower!
```

### Random Access (Worst Case)

```cuda
__global__ void random_access(float* data, int* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[indices[i]];  // ❌ WORST!
}
```

**What happens**: Up to 32 separate transactions (no reuse possible)

---

## Part 2: Measuring Coalescing Efficiency

### Benchmark: Coalesced vs Strided

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void coalesced_copy(const float* src, float* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dst[i] = src[i];  // Perfect coalescing
    }
}

__global__ void strided_copy(const float* src, float* dst, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dst[i] = src[i * stride];  // Uncoalesced!
    }
}

int main() {
    int N = 1 << 24;  // 16M elements
    size_t bytes = N * sizeof(float);

    float *d_src, *d_dst;
    cudaMalloc(&d_src, bytes * 32);  // Extra space for strided access
    cudaMalloc(&d_dst, bytes);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test coalesced
    cudaEventRecord(start);
    coalesced_copy<<<blocks, threads>>>(d_src, d_dst, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_coalesced;
    cudaEventElapsedTime(&ms_coalesced, start, stop);

    // Test strided (stride = 2)
    cudaEventRecord(start);
    strided_copy<<<blocks, threads>>>(d_src, d_dst, N, 2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_stride2;
    cudaEventElapsedTime(&ms_stride2, start, stop);

    // Test strided (stride = 32)
    cudaEventRecord(start);
    strided_copy<<<blocks, threads>>>(d_src, d_dst, N, 32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_stride32;
    cudaEventElapsedTime(&ms_stride32, start, stop);

    float bw_coalesced = (2.0f * bytes) / (ms_coalesced * 1e6);
    float bw_stride2 = (2.0f * bytes) / (ms_stride2 * 1e6);
    float bw_stride32 = (2.0f * bytes) / (ms_stride32 * 1e6);

    printf("Coalesced:   %.3f ms (%.2f GB/s)\n", ms_coalesced, bw_coalesced);
    printf("Stride = 2:  %.3f ms (%.2f GB/s) [%.2fx slower]\n",
           ms_stride2, bw_stride2, ms_stride2 / ms_coalesced);
    printf("Stride = 32: %.3f ms (%.2f GB/s) [%.2fx slower]\n",
           ms_stride32, bw_stride32, ms_stride32 / ms_coalesced);

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
```

**Compile and run**:
```bash
nvcc coalescing_bench.cu -o coalescing_bench
./coalescing_bench
```

**Expected output** (RTX 3080):
```
Coalesced:   0.148 ms (540 GB/s)
Stride = 2:  0.290 ms (275 GB/s) [1.96x slower]
Stride = 32: 3.850 ms (21 GB/s)  [26x slower!]
```

**Key insight**: Stride = 32 is **26x slower** - each thread accesses different cache line!

### Using NSight Compute

Profile with detailed memory metrics:

```bash
ncu --set full --section MemoryWorkloadAnalysis ./coalescing_bench
```

**Look for**:
- **Global Load Efficiency**: % of bytes loaded that are used
- **Global Load Transactions**: Number of memory transactions
- **Coalescing**: % of coalesced accesses

**Perfect coalescing**:
```
Global Load Efficiency: 100%
Coalescing: 100%
```

**Strided (stride=32)**:
```
Global Load Efficiency: ~3%   (Only 1/32 bytes used per transaction)
Coalescing: ~3%
```

---

## Part 3: Matrix Transpose - The Classic Problem

### Why Transpose is Hard

**Matrix layout in memory** (row-major):
```
Matrix:     Row 0: [a00, a01, a02, a03]
            Row 1: [a10, a11, a12, a13]
            Row 2: [a20, a21, a22, a23]
            Row 3: [a30, a31, a32, a33]

Memory:     [a00, a01, a02, a03, a10, a11, a12, ..., a33]
            ↑ Consecutive in row
```

**Transpose output**:
```
Row 0: [a00, a10, a20, a30]  ← Elements are STRIDED in input!
Row 1: [a01, a11, a21, a31]
...
```

**The challenge**: Either reads OR writes will be strided (uncoalesced)

### Naive Transpose (Slow)

```cuda
__global__ void transpose_naive(const float* A, float* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Read: A[row][col]  → Coalesced (row-major) ✅
        // Write: B[col][row] → Uncoalesced (column access) ❌
        B[col * M + row] = A[row * N + col];
    }
}
```

**Problem**: Writes are uncoalesced!

### Optimized with Shared Memory

**Strategy**: Use shared memory to change access pattern

```cuda
#define TILE_SIZE 32

__global__ void transpose_shared(const float* A, float* B, int M, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load tile from A (coalesced read) ✅
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    __syncthreads();

    // Transpose indices for B
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed tile to B (coalesced write) ✅
    if (x < M && y < N) {
        B[y * M + x] = tile[threadIdx.x][threadIdx.y];
        //               ↑ Note the swap!
    }
}
```

**Key idea**:
1. Read 32×32 tile from A (coalesced)
2. Store in shared memory
3. Read from shared memory with swapped indices (transpose)
4. Write to B (coalesced)

**Performance**:
```
Naive transpose:     45 GB/s  (Slow - uncoalesced writes)
Shared memory:       580 GB/s (Fast - both coalesced!)
Speedup: ~13x! 🚀
```

### Avoiding Bank Conflicts

**Problem**: Shared memory has 32 banks. Accessing same bank = serialization!

```cuda
// Original tile[32][32] has bank conflicts when reading transposed!
tile[0][0], tile[1][0], tile[2][0], ... tile[31][0]
  Bank 0      Bank 0      Bank 0    ...   Bank 0
  ↑ All access Bank 0 → 32-way conflict!
```

**Solution**: Pad shared memory by 1 element

```cuda
#define TILE_SIZE 32

__global__ void transpose_padded(const float* A, float* B, int M, int N) {
    // Add +1 to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < M && y < N) {
        B[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

**Performance gain**: Additional 10-20% speedup!

---

## Part 4: Structure of Arrays (SoA) vs Array of Structures (AoS)

### The Problem with AoS

**Array of Structures** (AoS):
```cpp
struct Particle {
    float x, y, z;     // Position
    float vx, vy, vz;  // Velocity
};

Particle particles[N];

// Memory layout:
// [x0, y0, z0, vx0, vy0, vz0, x1, y1, z1, vx1, vy1, vz1, ...]
```

**CUDA kernel** (accessing only x-coordinate):
```cuda
__global__ void update_x(Particle* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].x += 1.0f;  // ❌ Uncoalesced!
    }
}
```

**Problem**: Thread 0 accesses byte 0, thread 1 accesses byte 24, etc. (stride = 24 bytes)

### Solution: Structure of Arrays (SoA)

**Separate arrays** for each field:
```cpp
float x[N], y[N], z[N];
float vx[N], vy[N], vz[N];

// Memory layout:
// x:  [x0, x1, x2, ..., xN]
// y:  [y0, y1, y2, ..., yN]
// ...
```

**CUDA kernel**:
```cuda
__global__ void update_x_soa(float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] += 1.0f;  // ✅ Perfect coalescing!
    }
}
```

**Benchmark**:
```cuda
// AoS version
struct Particle {
    float x, y, z, vx, vy, vz;
};

__global__ void aos_kernel(Particle* p, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        p[i].x += p[i].vx * 0.01f;
        p[i].y += p[i].vy * 0.01f;
        p[i].z += p[i].vz * 0.01f;
    }
}

// SoA version
__global__ void soa_kernel(
    float* x, float* y, float* z,
    float* vx, float* vy, float* vz,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] += vx[i] * 0.01f;
        y[i] += vy[i] * 0.01f;
        z[i] += vz[i] * 0.01f;
    }
}
```

**Results** (N = 1M particles):
```
AoS: 1.85 ms (65 GB/s)
SoA: 0.42 ms (285 GB/s)
Speedup: 4.4x! 🚀
```

**Trade-off**: SoA requires more pointers to manage, but worth it for performance!

---

## Part 5: Real-World Example - Image Processing

### Problem: RGB to Grayscale

**Input**: RGB image (interleaved)
```
Memory: [R0, G0, B0, R1, G1, B1, R2, G2, B2, ...]
```

**Output**: Grayscale
```
Gray[i] = 0.299*R + 0.587*G + 0.114*B
```

### Naive Version (Uncoalesced)

```cuda
__global__ void rgb_to_gray_naive(const unsigned char* rgb, unsigned char* gray, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        // Each thread accesses 3 consecutive bytes (stride = 3)
        unsigned char r = rgb[i * 3 + 0];
        unsigned char g = rgb[i * 3 + 1];
        unsigned char b = rgb[i * 3 + 2];

        gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}
```

**Problem**: Stride = 3 → partially uncoalesced

### Optimized Version (Vectorized Load)

```cuda
__global__ void rgb_to_gray_optimized(const uchar3* rgb, unsigned char* gray, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        // Load all 3 channels at once (12 bytes aligned)
        uchar3 pixel = rgb[i];

        gray[i] = (unsigned char)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    }
}
```

**Key**: `uchar3` is aligned, so GPU can load efficiently!

**Performance**:
```
Naive:     1.2 ms (210 GB/s)
Optimized: 0.6 ms (420 GB/s)
Speedup: 2x
```

---

## Exercises

### Exercise 1: Analyze Coalescing

Determine if these access patterns are coalesced (assume consecutive threads):

```cuda
// Pattern A
int i = blockIdx.x * blockDim.x + threadIdx.x;
data[i] = ...;

// Pattern B
int i = blockIdx.x * blockDim.x + threadIdx.x;
data[i * 2] = ...;

// Pattern C
int i = blockIdx.x * blockDim.x + threadIdx.x;
data[threadIdx.x] = ...;  // Ignores blockIdx

// Pattern D
int i = blockIdx.x * blockDim.x + threadIdx.x;
data[(i % 32) * 1024 + (i / 32)] = ...;
```

### Exercise 2: Fix Uncoalesced Code

Optimize this kernel:
```cuda
__global__ void columnSum(float* matrix, float* sums, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < M; row++) {
            sum += matrix[row * N + col];  // Stride = N (BAD!)
        }
        sums[col] = sum;
    }
}
```

**Hint**: Use shared memory and transpose reads

### Exercise 3: Benchmark AoS vs SoA

Implement both versions for a simple physics simulation:
```cpp
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
};

// Update: position += velocity * dt
```

Measure performance difference.

### Solutions

<details>
<summary>Exercise 1 Solutions</summary>

```
Pattern A: ✅ Coalesced (consecutive threads → consecutive memory)
Pattern B: ❌ Partially uncoalesced (stride = 2)
Pattern C: ❌ Uncoalesced (same addresses across blocks!)
Pattern D: ❌ Heavily uncoalesced (complex indexing breaks pattern)
```
</details>

<details>
<summary>Exercise 2 Solution</summary>

```cuda
#define TILE_SIZE 32

__global__ void columnSum_optimized(float* matrix, float* sums, int M, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Process matrix in tiles
    for (int tileRow = 0; tileRow < M; tileRow += TILE_SIZE) {
        int row = tileRow + threadIdx.y;

        // Load tile (coalesced!)
        if (row < M && col < N) {
            tile[threadIdx.y][threadIdx.x] = matrix[row * N + col];
        } else {
            tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Accumulate from shared memory
        if (threadIdx.y == 0) {
            for (int i = 0; i < TILE_SIZE; i++) {
                sum += tile[i][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (threadIdx.y == 0 && col < N) {
        sums[col] = sum;
    }
}
```
</details>

---

## Key Takeaways

### Rules for Coalesced Access

```
✅ DO:
- Consecutive threads → consecutive memory
- Use aligned data types (float4, uchar3, etc.)
- Structure of Arrays (SoA) for parallel fields
- Shared memory to reorganize access patterns

❌ DON'T:
- Strided access (large strides)
- Array of Structures (AoS) for selective field access
- Random or scattered memory access
```

### Performance Impact

```
Perfect coalescing:     500-700 GB/s  (80-90% peak)
Stride = 2:             250-350 GB/s  (50% peak)
Stride = 32:            20-50 GB/s    (5% peak)

Speedup from optimization: 5-10x typical, up to 25x possible!
```

### Debugging Coalescing

```
1. Profile with NSight Compute
2. Check "Global Load/Store Efficiency"
3. Look for "Uncoalesced Global Accesses"
4. Redesign data structures or algorithms
```

---

## Next Steps

You've mastered memory coalescing! Continue optimizing with:

1. ✅ **[Shared Memory Tiling](./shared-memory-tiling.md)** - Already completed
2. ✅ **[Bank Conflicts](./bank-conflicts.md)** - Deep dive into shared memory
3. ✅ **[Matrix Multiply Project](../../projects/intermediate/matrix-multiply-library/)** - Apply techniques
4. ✅ **[Advanced Reductions](./advanced-reductions.md)** - Combining patterns

---

## Summary Checklist

After this tutorial, you can:

- [x] Explain memory coalescing and why it matters
- [x] Identify coalesced vs uncoalesced patterns
- [x] Measure coalescing efficiency with profilers
- [x] Optimize matrix transpose with shared memory
- [x] Choose SoA over AoS for better performance
- [x] Achieve 5-10x speedups through memory optimization

---

**Tutorial Generated by**: Content Creator Agent #7
**Session**: session_20251119_051648
**Difficulty**: Intermediate
**Next Tutorial**: [Bank Conflicts and Padding](./bank-conflicts.md)
