# Tutorial: Your First CUDA Kernel

## Learning Objectives

By the end of this tutorial, you will:
- ✅ Understand what a CUDA kernel is
- ✅ Write and compile your first CUDA program
- ✅ Launch a kernel from host code
- ✅ Transfer data between CPU and GPU
- ✅ Verify your kernel's correctness

**Level**: L0-L1 (Foundation/Beginner)
**Time**: 45-60 minutes
**Prerequisites**: Basic C/C++ knowledge, CUDA Toolkit installed

---

## Introduction

###  What is a CUDA Kernel?

A **kernel** is a function that runs on the GPU. Unlike regular C++ functions that run once, kernels are executed by **thousands of threads in parallel**.

**Analogy**: Think of a regular function as one worker doing a job sequentially. A kernel is like hiring 1,000 workers who all do the same task on different pieces of data simultaneously!

### Why Parallel Execution?

**Example Task**: Add 1 million numbers to another 1 million numbers

**CPU (Sequential)**:
```cpp
for (int i = 0; i < 1000000; i++) {
    c[i] = a[i] + b[i];  // One at a time
}
// Time: ~1-2 ms
```

**GPU (Parallel)**:
```cuda
// Launch 1 million threads, each does ONE addition
c[tid] = a[tid] + b[tid];  // All at once!
// Time: ~0.05 ms → 20-40x faster!
```

---

## Part 1: Vector Addition (The "Hello World" of CUDA)

We'll implement: `C[i] = A[i] + B[i]` for arrays of size N

### Step 1: The CUDA Kernel

```cuda
// vector_add.cu

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel: Runs on GPU
// __global__ means "callable from CPU, runs on GPU"
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // Calculate this thread's index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check (threads may exceed array size)
    if (i < N) {
        C[i] = A[i] + B[i];  // Each thread does ONE addition
    }
}
```

**Key Components**:
- `__global__`: Kernel qualifier (runs on GPU, called from CPU)
- `blockIdx.x`: Which block this thread belongs to
- `blockDim.x`: Number of threads per block
- `threadIdx.x`: Thread's position within its block

**Thread Index Calculation**:
```
Block 0: threads 0-255
Block 1: threads 256-511
Block 2: threads 512-767
...

Thread i in block b = b * block_size + i
```

### Step 2: Host Code (CPU)

```cuda
int main() {
    // Array size
    int N = 1 << 20;  // 1 Million elements
    size_t bytes = N * sizeof(float);

    // Allocate host (CPU) memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("Launching kernel: %d blocks, %d threads/block\n", blocks, threads_per_block);

    vectorAdd<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, h_C[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("✓ SUCCESS! All values correct.\n");
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

### Step 3: Compilation

```bash
# Compile
nvcc vector_add.cu -o vector_add

# Run
./vector_add
```

**Expected Output**:
```
Launching kernel: 4096 blocks, 256 threads/block
✓ SUCCESS! All values correct.
```

---

## Understanding the Execution

### Memory Flow

```
CPU (Host)                       GPU (Device)
┌──────────┐                     ┌──────────┐
│  h_A[]   │ ──cudaMemcpy──────> │  d_A[]   │
│  h_B[]   │ ──cudaMemcpy──────> │  d_B[]   │
└──────────┘                     └──────────┘
                                      │
                                  [Kernel]
                                      │
                                      ▼
                                 ┌──────────┐
                                 │  d_C[]   │
                                 └──────────┘
                                      │
┌──────────┐                          │
│  h_C[]   │ <──cudaMemcpy────────────┘
└──────────┘
```

### Kernel Launch Syntax

```cuda
kernel_name<<<blocks, threads_per_block>>>(args...);
```

**Translation for our example**:
```cuda
vectorAdd<<<4096, 256>>>(d_A, d_B, d_C, N);
```

Means:
- Launch 4,096 blocks
- Each block has 256 threads
- Total: 4,096 × 256 = 1,048,576 threads
- Perfect for our 1M element array!

### Thread Indexing Explained

For our launch: `<<<4096, 256>>>`

```
Block 0:        Block 1:        Block 2:
┌─────────┐    ┌─────────┐     ┌─────────┐
│ T0: i=0 │    │T0: i=256│     │T0: i=512│
│ T1: i=1 │    │T1: i=257│     │T1: i=513│
│   ...   │    │   ...   │     │   ...   │
│T255:255 │    │T255:511 │     │T255:767 │
└─────────┘    └─────────┘     └─────────┘

i = blockIdx.x * 256 + threadIdx.x
```

**Example calculations**:
- Thread 0 in Block 0: i = 0 * 256 + 0 = 0
- Thread 5 in Block 0: i = 0 * 256 + 5 = 5
- Thread 0 in Block 1: i = 1 * 256 + 0 = 256
- Thread 100 in Block 3: i = 3 * 256 + 100 = 868

---

## Part 2: Adding Error Checking

Always check for CUDA errors in real code!

```cuda
// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_A, bytes));
CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

// Check kernel launch
vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
CUDA_CHECK(cudaGetLastError());  // Check for launch errors
CUDA_CHECK(cudaDeviceSynchronize());  // Check for execution errors
```

---

## Part 3: Measuring Performance

Let's benchmark our kernel:

```cuda
#include <chrono>

// ... previous code ...

// Warm-up (important for accurate timing)
vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
CUDA_CHECK(cudaDeviceSynchronize());

// Timing
auto start = std::chrono::high_resolution_clock::now();

vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
CUDA_CHECK(cudaDeviceSynchronize());

auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> elapsed = end - start;

// Calculate bandwidth
double bandwidth_GB = (3.0 * N * sizeof(float)) / (elapsed.count() * 1e6);  // GB/s

printf("Time: %.3f ms\n", elapsed.count());
printf("Bandwidth: %.2f GB/s\n", bandwidth_GB);
```

**Expected Performance** (depends on GPU):
- RTX 3080: ~400-500 GB/s
- A100: ~1,000-1,200 GB/s

---

## Common Mistakes and How to Fix Them

### Mistake 1: Forgetting Boundary Check

```cuda
// ❌ WRONG: May access out of bounds
__global__ void bad(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];  // What if i >= N?
}

// ✅ CORRECT: Always check bounds
__global__ void good(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {  // Boundary check!
        C[i] = A[i] + B[i];
    }
}
```

### Mistake 2: Wrong Memory Transfer Direction

```cuda
// ❌ WRONG: Trying to use host pointer on device
vectorAdd<<<blocks, threads>>>(h_A, h_B, h_C, N);  // h_A is on CPU!

// ✅ CORRECT: Use device pointers
vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);  // d_A is on GPU
```

### Mistake 3: Not Synchronizing

```cuda
// ❌ WRONG: Kernel hasn't finished yet!
vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);  // Race condition!

// ✅ CORRECT: Wait for kernel
vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();  // Wait!
cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);  // Now safe
```

---

## Exercises

### Exercise 1: Vector Scaling
Modify the kernel to compute `C[i] = A[i] * scalar` where scalar is 2.5.

**Hint**: Add a scalar parameter to the kernel.

### Exercise 2: Element-wise Multiply
Create a kernel that computes `C[i] = A[i] * B[i]`.

**Hint**: Change one character in the kernel!

### Exercise 3: Array Initialization
Write a kernel that initializes an array: `A[i] = i * i`.

**Hint**: You only need one input array (the output).

### Solutions

<details>
<summary>Click to reveal solutions</summary>

```cuda
// Exercise 1: Vector Scaling
__global__ void vectorScale(const float* A, float* C, float scalar, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] * scalar;
    }
}

// Exercise 2: Element-wise Multiply
__global__ void vectorMul(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] * B[i];  // Changed + to *
    }
}

// Exercise 3: Array Initialization
__global__ void initArray(float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = i * i;
    }
}
```
</details>

---

## Next Steps

Congratulations! You've written your first CUDA kernel. Next:

1. ✅ **Complete**: `tutorials/beginner/thread-hierarchy.md`
   - Learn about 2D and 3D grids/blocks
   - Understand thread organization

2. ✅ **Complete**: `tutorials/beginner/memory-basics.md`
   - Learn about different memory types
   - Understand memory bandwidth

3. ✅ **Project**: `projects/beginner/activation-library/`
   - Build a complete activation function library
   - Practice what you learned

---

## Summary

**Key Concepts**:
- ✅ Kernels run on GPU, launched from CPU
- ✅ Each thread has a unique ID: `blockIdx.x * blockDim.x + threadIdx.x`
- ✅ Always check boundaries: `if (i < N)`
- ✅ Memory flow: CPU → GPU → Kernel → GPU → CPU
- ✅ Use `cudaDeviceSynchronize()` before reading results

**Typical CUDA Program Flow**:
```
1. Allocate host memory
2. Initialize data
3. Allocate device memory
4. Copy data to device
5. Launch kernel
6. Synchronize
7. Copy results back
8. Verify and use results
9. Free memory
```

---

**Tutorial Generated by**: Content Creator Agent #4
**Session**: session_20251119_051648
**Difficulty**: Beginner
**Next Tutorial**: [Thread Hierarchy Deep Dive](./thread-hierarchy.md)
