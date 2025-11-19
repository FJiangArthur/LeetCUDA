# Tutorial: CUDA Memory Model Basics

## Learning Objectives

By the end of this tutorial, you will:
- ✅ Understand different memory types in CUDA (global, shared, local, constant)
- ✅ Allocate and transfer data between CPU and GPU
- ✅ Measure memory bandwidth and identify bottlenecks
- ✅ Choose appropriate memory types for different use cases
- ✅ Optimize memory access patterns

**Level**: L1 (Beginner)
**Time**: 60-75 minutes
**Prerequisites**: Your First CUDA Kernel, Thread Hierarchy basics

---

## Introduction

### Why Memory Matters

**Critical Insight**: For most CUDA kernels, **memory bandwidth is the bottleneck**, not compute power!

**Example Performance**:
- **RTX 3080**: 29,770 GFLOPS (compute) vs 760 GB/s (memory bandwidth)
- **A100**: 19,500 GFLOPS vs 1,935 GB/s

**What this means**: You can do ~40 floating-point operations while waiting for ONE memory fetch!

### The Memory Hierarchy

CUDA GPUs have multiple memory types, each with different characteristics:

```
Memory Type       | Size      | Latency | Bandwidth  | Scope
------------------|-----------|---------|------------|-------------
Registers         | 256 KB    | 1 cycle | ~20 TB/s   | Per thread
Shared Memory     | 48-164 KB | ~30 cycles | ~15 TB/s | Per block
L1 Cache          | 128 KB    | ~30 cycles | ~15 TB/s | Per SM
L2 Cache          | 6-40 MB   | ~200 cycles | ~4 TB/s | Device
Global Memory     | 8-80 GB   | ~300 cycles | ~1 TB/s | Device
Constant Memory   | 64 KB     | ~300 cycles* | ~1 TB/s* | Device
(*cached)
```

**Key Takeaway**: Faster memory is **much** smaller. Good CUDA code moves data through the hierarchy efficiently.

---

## Part 1: Global Memory - The Foundation

### What is Global Memory?

**Global memory** is:
- The main GPU RAM (VRAM)
- Largest memory space (GBs)
- Slowest to access (~300-600 cycles)
- Accessible from all threads
- Persists across kernel launches

### Allocating Global Memory

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    size_t N = 1 << 24;  // 16 Million elements
    size_t bytes = N * sizeof(float);

    // Host (CPU) memory
    float *h_data = (float*)malloc(bytes);

    // Device (GPU) memory
    float *d_data;
    cudaMalloc(&d_data, bytes);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.5f;
    }

    // Transfer: Host → Device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // [Kernel execution here]

    // Transfer: Device → Host
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

### Memory Transfer Directions

```cuda
// Four transfer types:
cudaMemcpy(dest, src, bytes, cudaMemcpyHostToHost);     // CPU → CPU
cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice);   // CPU → GPU
cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToHost);   // GPU → CPU
cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice); // GPU → GPU
```

---

## Part 2: Measuring Memory Bandwidth

### The Bandwidth Test

Let's measure actual memory bandwidth with a simple copy kernel:

```cuda
__global__ void memcpyKernel(const float* src, float* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dst[i] = src[i];  // Simple memory copy
    }
}

int main() {
    int N = 1 << 26;  // 64M elements = 256 MB
    size_t bytes = N * sizeof(float);

    float *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);

    // Warm-up
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    memcpyKernel<<<blocks, threads>>>(d_src, d_dst, N);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    memcpyKernel<<<blocks, threads>>>(d_src, d_dst, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Calculate bandwidth
    // We read N floats (256 MB) and write N floats (256 MB) = 512 MB total
    float bandwidth_GB = (2.0f * bytes) / (ms * 1e6);  // GB/s

    printf("Time: %.3f ms\n", ms);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_GB);

    // Get theoretical peak
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peak_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    printf("Theoretical Peak: %.2f GB/s\n", peak_bandwidth);
    printf("Efficiency: %.1f%%\n", 100.0f * bandwidth_GB / peak_bandwidth);

    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
```

**Expected Results**:
- **RTX 3080**: 600-700 GB/s (~80-90% efficiency)
- **A100**: 1,400-1,600 GB/s (~80-85% efficiency)

### Why Not 100% Efficiency?

Real-world factors limiting bandwidth:
1. **Memory access patterns** (we'll optimize in later tutorials)
2. **Cache behavior**
3. **Memory controller scheduling**
4. **PCIe overhead** (for transfers to/from CPU)

---

## Part 3: Memory Types in Detail

### 1. Registers (Fastest, Smallest)

Registers are **automatic** - any variable in a kernel:

```cuda
__global__ void useRegisters(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // These are stored in REGISTERS (very fast!)
    float a = data[i];
    float b = a * 2.0f;
    float c = b + 1.0f;

    data[i] = c;
}
```

**Characteristics**:
- ✅ Fastest access (1 cycle)
- ✅ Automatic allocation
- ✅ Private to each thread
- ❌ Limited quantity (~256 KB per SM / 1024-2048 threads = ~128-256 bytes per thread)

**Warning**: Using too many registers reduces **occupancy** (fewer active threads).

### 2. Shared Memory (Fast, Small, Manual)

Shared memory is **explicitly declared**:

```cuda
__global__ void useSharedMemory(float* data, int N) {
    // Declare shared memory (shared by all threads in block)
    __shared__ float sharedData[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load from global → shared (slow)
    if (i < N) {
        sharedData[tid] = data[i];
    }
    __syncthreads();  // Wait for all threads to load

    // Now access shared memory (fast!)
    float value = sharedData[tid];
    // ... do computations ...

    __syncthreads();  // Sync before writing back

    // Write back to global
    if (i < N) {
        data[i] = value;
    }
}
```

**Characteristics**:
- ✅ ~100x faster than global memory
- ✅ Shared within a block
- ✅ Perfect for thread cooperation
- ❌ Small size (48-164 KB per block)
- ❌ Requires explicit synchronization

**Use Cases**:
- Tiling algorithms (matrix multiply, convolution)
- Reduction operations
- Temporary storage for frequently accessed data

### 3. Constant Memory (Read-Only, Cached)

For data that **never changes** during kernel execution:

```cuda
__constant__ float constData[1024];  // Declared at file scope

// Host code to initialize
float h_data[1024] = { /* ... */ };
cudaMemcpyToSymbol(constData, h_data, sizeof(h_data));

__global__ void useConstantMemory(float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Read from constant memory (cached, fast if all threads read same value)
        output[i] = constData[i % 1024] * 2.0f;
    }
}
```

**Characteristics**:
- ✅ Cached (fast when all threads access same location)
- ✅ Read-only (safe for concurrent access)
- ❌ Limited size (64 KB)
- ❌ Slow if threads access different locations (serialized access)

**Best For**: Lookup tables, configuration data, coefficients

### 4. Local Memory (Actually Global, Per-Thread)

**Tricky concept**: "Local memory" is **not fast** - it's just **private global memory**.

```cuda
__global__ void spillsToLocal() {
    // If you use too many variables, they "spill" to local memory
    float bigArray[1000];  // Likely in local memory (too big for registers)

    // This is SLOW despite being "local" to the thread!
    for (int i = 0; i < 1000; i++) {
        bigArray[i] = i * 2.0f;  // Global memory access!
    }
}
```

**When it happens**:
- Arrays indexed with non-constant values
- Too many variables (register spilling)
- Large per-thread data structures

**How to avoid**: Use smaller arrays, shared memory, or global memory explicitly.

---

## Part 4: PCIe Transfer Optimization

### The PCIe Bottleneck

**Problem**: CPU ↔ GPU transfers are **slow** (16-32 GB/s vs GPU's 500-2000 GB/s)

```
GPU Global Memory: 760 GB/s (RTX 3080)
           ↕
    PCIe 4.0 x16: ~32 GB/s  ← BOTTLENECK!
           ↕
      CPU RAM: 50-100 GB/s
```

### Optimization Strategy 1: Minimize Transfers

```cuda
// ❌ BAD: Multiple small transfers
for (int i = 0; i < 1000; i++) {
    cudaMemcpy(d_data + i, h_data + i, sizeof(float), cudaMemcpyHostToDevice);
    kernel<<<1, 256>>>(d_data + i);
    cudaMemcpy(h_result + i, d_result + i, sizeof(float), cudaMemcpyDeviceToHost);
}

// ✅ GOOD: One large transfer
cudaMemcpy(d_data, h_data, 1000 * sizeof(float), cudaMemcpyHostToDevice);
kernel<<<blocks, 256>>>(d_data);
cudaMemcpy(h_result, d_result, 1000 * sizeof(float), cudaMemcpyDeviceToHost);
```

**Speedup**: 50-100x faster!

### Optimization Strategy 2: Pinned Memory

Regular malloc'd memory is **pageable** (can be swapped to disk). This forces extra copies.

```cuda
// ❌ Regular malloc (pageable)
float* h_data = (float*)malloc(bytes);
// Transfer requires: Pageable RAM → Pinned buffer → GPU (2 copies!)

// ✅ Pinned (page-locked) memory
float* h_data_pinned;
cudaMallocHost(&h_data_pinned, bytes);  // or cudaHostAlloc()
// Transfer: Pinned RAM → GPU (1 copy!)
```

**Benchmark**:
```cuda
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    size_t bytes = 256 * 1024 * 1024;  // 256 MB

    // Pageable memory
    float* h_pageable = (float*)malloc(bytes);

    // Pinned memory
    float* h_pinned;
    cudaMallocHost(&h_pinned, bytes);

    float* d_data;
    cudaMalloc(&d_data, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test pageable
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_pageable;
    cudaEventElapsedTime(&ms_pageable, start, stop);

    // Test pinned
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_pinned;
    cudaEventElapsedTime(&ms_pinned, start, stop);

    printf("Pageable: %.2f ms (%.2f GB/s)\n",
           ms_pageable, bytes / (ms_pageable * 1e6));
    printf("Pinned:   %.2f ms (%.2f GB/s)\n",
           ms_pinned, bytes / (ms_pinned * 1e6));
    printf("Speedup: %.2fx\n", ms_pageable / ms_pinned);

    free(h_pageable);
    cudaFreeHost(h_pinned);
    cudaFree(d_data);

    return 0;
}
```

**Expected Results**: 1.5-2x speedup with pinned memory!

**Trade-off**: Pinned memory consumes system resources. Don't over-use it.

### Optimization Strategy 3: Asynchronous Transfers

Overlap transfers with computation using streams:

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Pipeline: Transfer batch 1, compute batch 0
cudaMemcpyAsync(d_data1, h_data1, bytes, cudaMemcpyHostToDevice, stream1);
kernel<<<blocks, threads, 0, stream2>>>(d_data0);  // Runs in parallel!

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

**We'll cover this in advanced tutorials!**

---

## Part 5: Measuring Memory Performance

### Tool 1: CUDA Events (Kernel Time Only)

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<blocks, threads>>>(d_data);
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
```

**Pros**: Precise GPU timing
**Cons**: Only measures kernel, not transfers

### Tool 2: NSight Systems (Full Timeline)

```bash
# Profile entire application
nsys profile --stats=true ./my_program

# View in GUI
nsys-ui report.nsys-rep
```

Shows:
- Kernel execution times
- Memory transfer times
- CPU-GPU synchronization
- Gaps and inefficiencies

### Tool 3: NSight Compute (Kernel Details)

```bash
# Profile specific kernel
ncu --set full ./my_program
```

Shows:
- Memory throughput achieved
- Memory bandwidth utilization
- Compute utilization
- Bottleneck analysis

---

## Part 6: Common Memory Patterns

### Pattern 1: Map (Element-wise)

Each thread processes one element independently:

```cuda
__global__ void map(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = sqrt(in[i]);  // Independent operation
    }
}
```

**Memory characteristics**:
- Each element read/written once
- Bandwidth-bound (memory is bottleneck)
- Arithmetic intensity: 1 FLOP / 2 memory accesses

### Pattern 2: Reduction (Sum, Max, etc.)

Multiple threads collaborate to compute aggregate:

```cuda
__global__ void reduce(const float* in, float* out, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load into shared memory
    sdata[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
```

**Memory characteristics**:
- Uses shared memory for speed
- Global memory read once, write once per block
- Much higher arithmetic intensity

### Pattern 3: Stencil (Neighbors)

Each output depends on neighboring inputs:

```cuda
__global__ void stencil(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < N - 1) {
        // 3-point stencil
        out[i] = 0.25f * in[i-1] + 0.5f * in[i] + 0.25f * in[i+1];
    }
}
```

**Memory characteristics**:
- Multiple reads per output (redundant)
- Opportunity for shared memory optimization
- Cache helps with spatial locality

---

## Exercises

### Exercise 1: Bandwidth Comparison
Measure and compare bandwidth for:
1. Host-to-Device transfer
2. Device-to-Host transfer
3. Device-to-Device copy (cudaMemcpy with cudaMemcpyDeviceToDevice)

**Question**: Which is slowest and why?

### Exercise 2: Pinned vs Pageable
Implement the pinned memory benchmark above. Test with different sizes: 1MB, 16MB, 256MB, 1GB.

**Question**: At what size does the benefit of pinned memory become significant?

### Exercise 3: Register Pressure
Write two kernels:
```cuda
__global__ void lowRegisters(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = data[i];
        data[i] = x * 2.0f + 1.0f;
    }
}

__global__ void highRegisters(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x0 = data[i], x1 = x0 * 2, x2 = x1 * 2, x3 = x2 * 2;
        float x4 = x3 * 2, x5 = x4 * 2, x6 = x5 * 2, x7 = x6 * 2;
        float x8 = x7 * 2, x9 = x8 * 2, x10 = x9 * 2, x11 = x10 * 2;
        data[i] = x11;
    }
}
```

Compile with `nvcc -Xptxas -v` to see register usage. Compare occupancy.

### Solutions

<details>
<summary>Click to reveal Exercise 1 solution</summary>

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    size_t bytes = 256 * 1024 * 1024;  // 256 MB

    float *h_data = (float*)malloc(bytes);
    float *d_data1, *d_data2;
    cudaMalloc(&d_data1, bytes);
    cudaMalloc(&d_data2, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // H2D
    cudaEventRecord(start);
    cudaMemcpy(d_data1, h_data, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_h2d;
    cudaEventElapsedTime(&ms_h2d, start, stop);

    // D2H
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_data1, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_d2h;
    cudaEventElapsedTime(&ms_d2h, start, stop);

    // D2D
    cudaEventRecord(start);
    cudaMemcpy(d_data2, d_data1, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_d2d;
    cudaEventElapsedTime(&ms_d2d, start, stop);

    printf("H2D: %.2f ms (%.2f GB/s)\n", ms_h2d, bytes/(ms_h2d*1e6));
    printf("D2H: %.2f ms (%.2f GB/s)\n", ms_d2h, bytes/(ms_d2h*1e6));
    printf("D2D: %.2f ms (%.2f GB/s)\n", ms_d2d, bytes/(ms_d2d*1e6));

    // Answer: D2H is usually slowest (PCIe asymmetry + driver overhead)

    free(h_data);
    cudaFree(d_data1);
    cudaFree(d_data2);
    return 0;
}
```
</details>

---

## Key Takeaways

### Memory Hierarchy Summary

```
FASTEST → SLOWEST
Registers (256 KB, 1 cycle)
   ↓
Shared Memory (48-164 KB, ~30 cycles)
   ↓
L1/L2 Cache (automatic)
   ↓
Global Memory (GBs, ~300 cycles)
   ↓
CPU RAM via PCIe (slowest, ~1000s of cycles)
```

### Best Practices

1. ✅ **Minimize CPU-GPU transfers** - they're expensive!
2. ✅ **Use pinned memory** for unavoidable transfers
3. ✅ **Batch operations** to amortize transfer costs
4. ✅ **Measure bandwidth** to identify bottlenecks
5. ✅ **Use shared memory** for frequently accessed data
6. ✅ **Keep data on GPU** between kernel launches when possible

### When is Memory the Bottleneck?

Your kernel is **memory-bound** if:
- Arithmetic intensity is low (few operations per byte)
- Achieved bandwidth is close to theoretical peak
- Increasing compute doesn't improve performance

**Solution**: Memory access optimizations (coalescing, tiling, fusion)

### When is Compute the Bottleneck?

Your kernel is **compute-bound** if:
- Arithmetic intensity is high (many operations per byte)
- Memory bandwidth utilization is low
- Performance scales with compute power

**Solution**: Algorithmic optimizations, faster math, Tensor Cores

---

## Next Steps

**You've mastered memory basics!** Next:

1. ✅ **[Memory Coalescing](../intermediate/memory-coalescing.md)** - Optimize access patterns
2. ✅ **[Shared Memory Deep Dive](../intermediate/shared-memory-tiling.md)** - Tiling strategies
3. ✅ **[Element-wise Operations](./element-wise-operations.md)** - Apply memory knowledge
4. ✅ **[Debugging and Profiling](./debugging-profiling.md)** - Find memory bugs

---

## Summary Checklist

After this tutorial, you should be able to:

- [x] Explain the CUDA memory hierarchy
- [x] Allocate and transfer global memory
- [x] Measure memory bandwidth
- [x] Use shared memory in kernels
- [x] Understand when to use constant memory
- [x] Optimize PCIe transfers with pinned memory
- [x] Identify memory-bound vs compute-bound kernels
- [x] Profile memory performance with NSight tools

---

**Tutorial Generated by**: Content Creator Agent #5
**Session**: session_20251119_051648
**Difficulty**: Beginner
**Next Tutorial**: [Memory Coalescing Optimization](../intermediate/memory-coalescing.md)
