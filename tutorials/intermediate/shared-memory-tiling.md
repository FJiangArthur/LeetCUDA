# Tutorial: Shared Memory and Tiling

## Learning Objectives

By the end of this tutorial, you will:
- ✅ Understand what shared memory is and why it's fast
- ✅ Implement tiling strategies for matrix operations
- ✅ Use `__syncthreads()` correctly
- ✅ Achieve 10-50x speedups through shared memory
- ✅ Avoid common shared memory pitfalls

**Level**: L2 (Intermediate)
**Time**: 90-120 minutes
**Prerequisites**: [Thread Hierarchy](../beginner/thread-hierarchy.md), [Memory Basics](../beginner/memory-basics.md)

---

## Introduction

### The Performance Problem

**Global memory is SLOW**:
- Latency: ~400-800 cycles
- Even with caching, bandwidth-limited kernels suffer

**Shared memory is FAST**:
- Latency: ~5-20 cycles (20-100x faster!)
- Located on-chip, near compute units
- BUT: Only ~48-100 KB per SM

**Key Insight**: Use shared memory as a user-managed cache!

---

## Part 1: Understanding Shared Memory

### What is Shared Memory?

**Definition**: Fast, on-chip memory shared among all threads in a block

```
SM (Streaming Multiprocessor)
┌─────────────────────────────────┐
│  CUDA Cores                      │
│  ┌─┐┌─┐┌─┐┌─┐                   │
│  └─┘└─┘└─┘└─┘  ...              │
│                                  │
│  Shared Memory (48-164 KB)       │
│  ┌──────────────────────────┐   │
│  │  Fast, On-Chip Memory    │   │
│  │  Shared by all threads   │   │
│  │  in the block            │   │
│  └──────────────────────────┘   │
│         ↑↑↑ 20-100x faster!      │
│                                  │
│  Global Memory Interface         │
│         ↓↓↓ slow!                │
└─────────────────────────────────┘
         ║
         ║ PCIe/NVLink
         ║
┌────────────────────┐
│  Global Memory     │
│  (GB-scale, slow)  │
└────────────────────┘
```

### Declaring Shared Memory

**Static Allocation**:
```cuda
__global__ void kernel() {
    // Fixed size, known at compile time
    __shared__ float sharedData[256];

    // Use it
    sharedData[threadIdx.x] = someValue;
}
```

**Dynamic Allocation**:
```cuda
__global__ void kernel() {
    // Size determined at launch time
    extern __shared__ float sharedData[];

    // Use it
    sharedData[threadIdx.x] = someValue;
}

// Launch with shared memory size
kernel<<<blocks, threads, sharedMemBytes>>>();
```

---

## Part 2: Simple Shared Memory Example

### Reverse Array in Block

**Goal**: Reverse elements within each block

```cuda
__global__ void reverseBlock(float* input, float* output, int N) {
    // Allocate shared memory
    __shared__ float temp[256];  // Assuming 256 threads/block

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load data to shared memory
    if (gid < N) {
        temp[tid] = input[gid];
    }

    // CRITICAL: Wait for all threads to finish loading
    __syncthreads();

    // Write reversed data to global memory
    if (gid < N) {
        output[gid] = temp[blockDim.x - 1 - tid];
    }
}
```

**Why __syncthreads()?**
```
Without sync:
Thread 0: Loads temp[0] ━━━━━━━┓
                                ├→ Reads temp[255] (NOT READY YET!)
Thread 255: Loads temp[255] ━━━┛

With sync:
Thread 0: Loads temp[0] ━━━┓
                           ├→ __syncthreads() ←┐
Thread 255: Loads temp[255]┛                   │
                                               │
All threads wait here ════════════════════════┘
                                               │
Thread 0: Reads temp[255] (NOW READY!) ←──────┘
```

---

## Part 3: Matrix Multiplication - The Tiling Pattern

### Naive Matrix Multiplication (Slow)

**Problem**: Multiply two N×N matrices C = A × B

```cuda
// Naive version: All global memory accesses
__global__ void matmulNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
            // ↑ Every iteration: 2 global memory reads!
        }
        C[row * N + col] = sum;
    }
}

// For N=1024: Each thread does 1024 global reads
// Performance: ~5-10% of peak (memory-bound!)
```

**Problem**: Each element of A and B is loaded N times from global memory!

---

### Tiled Matrix Multiplication (Fast!)

**Key Idea**:
1. Divide matrices into tiles that fit in shared memory
2. Load each tile once into shared memory
3. Reuse tile data for multiple computations
4. Dramatic reduction in global memory accesses!

**Algorithm**:
```
For each output element C[row][col]:
  sum = 0
  For each tile along the K dimension:
    1. Load tile of A and B into shared memory
    2. Sync threads
    3. Compute partial sum using shared memory
    4. Sync threads
  Write sum to C[row][col]
```

### Implementation

```cuda
#define TILE_SIZE 16

__global__ void matmulTiled(float* A, float* B, float* C, int N) {
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of C to compute
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Allocate shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int aRow = row;
        int aCol = t * TILE_SIZE + tx;
        if (aRow < N && aCol < N) {
            tileA[ty][tx] = A[aRow * N + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + ty;
        int bCol = col;
        if (bRow < N && bCol < N) {
            tileB[ty][tx] = B[bRow * N + bCol];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // Sync: Ensure tiles are loaded before computing
        __syncthreads();

        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
            // ↑ Shared memory reads: 20-100x faster than global!
        }

        // Sync: Ensure all threads done before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Visual Explanation

```
Matrix A (N×N)         Matrix B (N×N)         Matrix C (N×N)
┌──────────────┐      ┌──────────────┐       ┌──────────────┐
│              │      │              │       │              │
│  [Tile A0]───│──┐   │              │   ┌──→│   [Result]   │
│              │  │   │  [Tile B0]   │   │   │              │
│  [Tile A1]   │  │   │      ↓       │   │   │              │
│      .       │  │   │  [Tile B1]   │   │   │              │
│      .       │  │   │      ↓       │   │   │              │
│              │  │   │      .       │   │   │              │
└──────────────┘  │   └──────────────┘   │   └──────────────┘
                  │                       │
                  └→ Compute in ──────────┘
                     Shared Memory

Phase 1: Load Tile A0 and Tile B0 → Compute partial sum
Phase 2: Load Tile A1 and Tile B1 → Add to sum
...
Result: Full dot product for C element
```

### Performance Comparison

**For N=1024, TILE_SIZE=16**:

**Naive**:
- Global memory reads per thread: 2,048 (1024 from A + 1024 from B)
- Effective bandwidth: ~50 GB/s (5% of peak)

**Tiled**:
- Global memory reads per thread: 128 (64 from A + 64 from B)
- Reuse factor: 16× (each loaded element used 16 times)
- Effective bandwidth: ~600 GB/s (60% of peak)
- **Speedup: 12-15x faster!**

---

## Part 4: __syncthreads() - Critical Understanding

### What it Does

**Barrier synchronization** within a block:
```cuda
__syncthreads();
```

1. **Each thread** reaches this point
2. **Waits** until ALL threads in block arrive
3. **Then** all threads proceed together

### When to Use

**Rule 1**: After loading to shared memory, before reading
```cuda
// Load data
sharedMem[tid] = globalMem[gid];
__syncthreads();  // ← Essential!
// Now safe to read sharedMem[other_tid]
float value = sharedMem[tid + 1];
```

**Rule 2**: After computing, before next load (if reusing shared mem)
```cuda
// Compute using shared memory
result = sharedMem[tid] * 2;
__syncthreads();  // ← Essential before reusing!
// Now safe to load new data
sharedMem[tid] = newData;
```

### Common Mistakes

**Mistake 1**: Forgetting __syncthreads()
```cuda
// ❌ WRONG: Race condition!
__shared__ float data[256];
data[tid] = input[gid];
// Missing __syncthreads()!
float value = data[tid + 1];  // Might not be ready!
```

**Mistake 2**: Conditional __syncthreads()
```cuda
// ❌ WRONG: Deadlock if not all threads reach sync!
if (tid < 128) {
    __syncthreads();  // Only half of threads reach here!
}
// → Deadlock! Other threads waiting forever
```

**Correct**:
```cuda
// ✅ CORRECT: All threads must reach sync
__syncthreads();
if (tid < 128) {
    // Do work
}
```

---

## Part 5: Shared Memory Patterns

### Pattern 1: Reduction in Shared Memory

**Goal**: Sum all elements in a block

```cuda
__global__ void reduceBlock(float* input, float* output, int N) {
    __shared__ float sharedData[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load to shared memory
    sharedData[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
```

**Visualization**:
```
Initial: [a0, a1, a2, a3, a4, a5, a6, a7]

Stride 4:
Thread 0: a0 += a4  [a0+a4, a1, a2, a3, a4, a5, a6, a7]
Thread 1: a1 += a5  [a0+a4, a1+a5, a2, a3, a4, a5, a6, a7]
...
Result:  [a0+a4, a1+a5, a2+a6, a3+a7, a4, a5, a6, a7]
__syncthreads()

Stride 2:
Thread 0: a0 += a2  [a0+a2+a4+a6, ...]
Thread 1: a1 += a3  [a0+a2+a4+a6, a1+a3+a5+a7, ...]
__syncthreads()

Stride 1:
Thread 0: a0 += a1  [sum_all, ...]

Final: sharedData[0] = a0+a1+a2+a3+a4+a5+a6+a7
```

### Pattern 2: Prefix Sum (Scan)

**Goal**: Compute cumulative sum [a0, a0+a1, a0+a1+a2, ...]

```cuda
__global__ void scanBlock(float* input, float* output, int N) {
    __shared__ float temp[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load data
    temp[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    // Up-sweep phase (reduce)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Down-sweep phase
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            float t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }

    // Write result
    if (gid < N) {
        output[gid] = temp[tid];
    }
}
```

### Pattern 3: Halo/Ghost Cells (Stencil)

**Goal**: Apply 3-point stencil: `out[i] = in[i-1] + in[i] + in[i+1]`

```cuda
#define BLOCK_SIZE 256
#define RADIUS 1

__global__ void stencil1D(float* input, float* output, int N) {
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load main data
    temp[tid + RADIUS] = (gid < N) ? input[gid] : 0.0f;

    // Load halo cells (left and right neighbors)
    if (tid < RADIUS) {
        // Left halo
        int left_gid = blockIdx.x * blockDim.x - RADIUS + tid;
        temp[tid] = (left_gid >= 0) ? input[left_gid] : 0.0f;

        // Right halo
        int right_gid = blockIdx.x * blockDim.x + blockDim.x + tid;
        temp[tid + BLOCK_SIZE + RADIUS] = (right_gid < N) ? input[right_gid] : 0.0f;
    }

    __syncthreads();

    // Apply stencil using shared memory
    if (gid < N) {
        output[gid] = temp[tid] + temp[tid + RADIUS] + temp[tid + 2 * RADIUS];
    }
}
```

---

## Part 6: Performance Tips

### Tip 1: Minimize __syncthreads()

**Bad**:
```cuda
for (int i = 0; i < 100; i++) {
    __syncthreads();  // 100 syncs! Expensive!
    // work
}
```

**Good**:
```cuda
// Reorganize to reduce syncs
__syncthreads();
for (int i = 0; i < 100; i++) {
    // work (no dependencies between iterations)
}
```

### Tip 2: Avoid Bank Conflicts

(See [Bank Conflicts tutorial](./bank-conflicts.md) for details)

```cuda
// BAD: Stride access causes conflicts
__shared__ float data[32][32];
float value = data[threadIdx.x][0];  // All threads access column 0!

// GOOD: Coalesced access
float value = data[0][threadIdx.x];  // Threads access different banks
```

### Tip 3: Optimal Tile Size

**Trade-offs**:
- Larger tiles: Fewer global memory accesses, more reuse
- Smaller tiles: Fit more blocks on SM, better occupancy

**Sweet spots**:
- 16×16 = 256 threads (common)
- 32×32 = 1024 threads (max, less flexible)
- 8×8 = 64 threads (too small, low occupancy)

---

## Practice Exercises

### Exercise 1: Matrix Transpose with Shared Memory

Implement matrix transpose using tiling to avoid uncoalesced writes.

<details>
<summary>Hint</summary>

```cuda
// Load tile from input (coalesced)
// Store tile to output (also need to be coalesced!)
// Trick: Transpose in shared memory
```
</details>

### Exercise 2: 1D Convolution

Implement 1D convolution with kernel size 5 using shared memory.

<details>
<summary>Hint</summary>

```cuda
// Load block + 4 halo cells (2 left, 2 right)
// Apply 5-point filter in shared memory
```
</details>

### Exercise 3: Parallel Prefix Sum

Implement Hillis-Steele scan (simpler than work-efficient).

<details>
<summary>Hint</summary>

```cuda
for (int stride = 1; stride < n; stride *= 2) {
    if (tid >= stride) {
        temp[tid] = temp[tid] + temp[tid - stride];
    }
    __syncthreads();
}
```
</details>

---

## Summary

### Key Concepts

**Shared Memory**:
- ✅ 20-100x faster than global memory
- ✅ Shared among all threads in a block
- ✅ Limited size (~48-164 KB per SM)
- ✅ Requires explicit management

**Tiling**:
- ✅ Divide data into blocks that fit in shared memory
- ✅ Load once, reuse many times
- ✅ Dramatically reduces global memory traffic
- ✅ Key technique for high-performance kernels

**__syncthreads()**:
- ✅ Synchronizes all threads in a block
- ✅ Essential after loading, before reading
- ✅ Must be reached by ALL threads (no conditionals!)
- ✅ Avoid overuse (impacts performance)

### Performance Impact

Typical speedups with shared memory tiling:
- **Matrix multiplication**: 10-20x
- **Convolution**: 5-15x
- **Reduction**: 50-100x
- **Stencil operations**: 10-30x

---

## Next Steps

1. ✅ **Practice**: Implement exercises
2. ✅ **Advanced**: [Bank Conflicts](./bank-conflicts.md)
3. ✅ **Project**: [Optimized Convolution](../../projects/intermediate/convolution-engine/)
4. ✅ **Next**: [Advanced Reductions](./advanced-reductions.md)

---

**Tutorial Generated by**: Content Creator Agent #5
**Difficulty**: Intermediate (L2)
**Time**: 90-120 minutes
**Critical for**: All high-performance CUDA programming!
