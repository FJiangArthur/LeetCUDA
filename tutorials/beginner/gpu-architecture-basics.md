# Tutorial: Understanding GPU Architecture

## Learning Objectives

By the end of this tutorial, you will:
- ✅ Understand the fundamental differences between CPUs and GPUs
- ✅ Explain the SIMT (Single Instruction, Multiple Thread) execution model
- ✅ Identify when GPU acceleration is beneficial
- ✅ Understand GPU memory hierarchy
- ✅ Know key GPU architecture components

**Level**: L0 (Foundation)
**Time**: 45-60 minutes
**Prerequisites**: None (beginner-friendly)

---

## Introduction

### Why GPUs Exist

**The Problem**: CPUs are great at sequential tasks but struggle with massive parallelism.

**Example - Image Processing**:
```
Task: Apply filter to 4K image (3840×2160 = 8,294,400 pixels)

CPU Approach:
for (int i = 0; i < 8294400; i++) {
    output[i] = filter(input[i]);
}
Time: ~100 ms (processing one pixel at a time)

GPU Approach:
Launch 8,294,400 threads in parallel
Each thread: output[i] = filter(input[i]);
Time: ~2 ms (all pixels processed simultaneously)

Speedup: 50x faster!
```

---

## Part 1: CPU vs GPU - The Fundamental Difference

### CPU: The Generalist

**Architecture**:
```
CPU (e.g., Intel Core i9)
┌─────────────────────────────────┐
│  4-8 Powerful Cores             │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐      │
│  │ C │ │ C │ │ C │ │ C │      │
│  └───┘ └───┘ └───┘ └───┘      │
│                                 │
│  Large Cache (32MB L3)          │
│  ┌───────────────────────┐     │
│  │   Shared L3 Cache      │     │
│  └───────────────────────┘     │
│                                 │
│  Complex Control Logic          │
│  - Branch Prediction            │
│  - Out-of-order Execution      │
│  - Speculative Execution       │
└─────────────────────────────────┘
```

**Characteristics**:
- **Few cores** (4-16 typically): Each very powerful
- **High clock speed** (3-5 GHz): Fast sequential execution
- **Large cache** (MB-scale): Reduce memory latency
- **Complex control**: Branch prediction, out-of-order execution
- **Good at**: Sequential tasks, complex logic, unpredictable branches

**Analogy**: A team of 8 expert chefs who can handle any recipe complexity

---

### GPU: The Specialist

**Architecture**:
```
GPU (e.g., NVIDIA RTX 3080)
┌─────────────────────────────────────────────────────────┐
│  68 Streaming Multiprocessors (SMs)                     │
│  Each SM has 128 CUDA Cores = 8,704 total cores!       │
│                                                          │
│  SM 0        SM 1        SM 2        ...    SM 67       │
│  ┌────┐     ┌────┐     ┌────┐            ┌────┐       │
│  │128 │     │128 │     │128 │            │128 │       │
│  │cores     │cores     │cores            │cores│       │
│  └────┘     └────┘     └────┘            └────┘       │
│  ├─ Small Cache (128KB Shared Memory per SM)           │
│  ├─ Simple Control                                      │
│  └─ Optimized for Parallel Execution                    │
│                                                          │
│  Memory: 10GB GDDR6X (High Bandwidth: 760 GB/s)        │
└─────────────────────────────────────────────────────────┘
```

**Characteristics**:
- **Many cores** (1,000s-10,000s): Each simpler than CPU core
- **Lower clock speed** (1-2 GHz): But massive parallelism compensates
- **Smaller cache per core**: Relies on high memory bandwidth
- **Simple control**: Minimal branch prediction
- **Good at**: Parallel tasks, regular patterns, data parallelism

**Analogy**: A factory with 8,000 workers doing simple, repetitive tasks simultaneously

---

### When to Use GPU vs CPU

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| **Process 1M numbers** | GPU | Parallel, same operation on all |
| **Sort complex database** | CPU | Irregular access patterns |
| **Matrix multiplication** | GPU | Highly parallel, regular pattern |
| **Parse JSON file** | CPU | Irregular, branching logic |
| **Image convolution** | GPU | Same filter applied everywhere |
| **Web server request** | CPU | I/O bound, irregular |
| **Deep learning training** | GPU | Massive matrix ops, parallelism |
| **Compile code** | CPU | Sequential dependencies |

**Rule of Thumb**:
- **GPU**: Same operation on lots of data (data parallelism)
- **CPU**: Different operations, complex logic, irregular patterns

---

## Part 2: SIMT - How GPUs Execute Code

### SIMT: Single Instruction, Multiple Thread

**Key Concept**: GPUs execute the **same instruction** on **many threads** simultaneously.

**Example**:
```cuda
// CUDA kernel
__global__ void add(float* a, float* b, float* c, int N) {
    int i = threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];  // Same instruction, different data
    }
}

// Execution with 1024 threads:
Thread 0: c[0] = a[0] + b[0]  ┐
Thread 1: c[1] = a[1] + b[1]  │
Thread 2: c[2] = a[2] + b[2]  │ All execute
...                           │ simultaneously!
Thread 1023: c[1023] = a[1023] + b[1023] ┘
```

### Warp: The Execution Unit

**Warp**: Group of 32 threads that execute together in lockstep

```
Block of 128 threads = 4 warps
┌──────────────┐
│ Warp 0       │ Threads 0-31   execute together
├──────────────┤
│ Warp 1       │ Threads 32-63  execute together
├──────────────┤
│ Warp 2       │ Threads 64-95  execute together
├──────────────┤
│ Warp 3       │ Threads 96-127 execute together
└──────────────┘
```

**Important**: All 32 threads in a warp execute the same instruction at the same time!

---

### Branch Divergence: The Enemy of Performance

**Problem**: When threads in a warp take different paths

```cuda
// BAD: Branch divergence
__global__ void divergent(int* data) {
    int i = threadIdx.x;
    if (data[i] > 0) {      // Some threads go here
        data[i] = data[i] * 2;
    } else {                // Others go here
        data[i] = data[i] * 3;
    }
}

// What actually happens in hardware:
Warp executes:
1. Threads with data[i] > 0: execute multiply by 2, others IDLE
2. Threads with data[i] <= 0: execute multiply by 3, others IDLE
Result: Serialization! Performance loss!
```

**Good Practice**: Minimize divergence
```cuda
// BETTER: No divergence (if possible)
__global__ void no_divergence(int* data) {
    int i = threadIdx.x;
    // Same path for all threads
    int multiplier = (data[i] > 0) ? 2 : 3;
    data[i] = data[i] * multiplier;
    // Still has conditional, but in arithmetic, not control flow
}
```

---

## Part 3: GPU Memory Hierarchy

### Memory Types (Fastest to Slowest)

```
┌─────────────────────────────────────────────────────┐
│ 1. Registers (Per-thread, private)                  │
│    - Fastest (~1 cycle latency)                     │
│    - Tiny (~256 KB total per SM, divided among      │
│      all threads)                                    │
│    - Automatic variables in kernel                  │
└─────────────────────────────────────────────────────┘
                      ↓ 10x slower
┌─────────────────────────────────────────────────────┐
│ 2. Shared Memory (Per-block, shared)                │
│    - Very fast (~5-20 cycles)                       │
│    - Small (~100-200 KB per SM)                     │
│    - Explicitly managed by programmer               │
│    - Shared among threads in a block                │
└─────────────────────────────────────────────────────┘
                      ↓ 5x slower
┌─────────────────────────────────────────────────────┐
│ 3. L1/L2 Cache (Automatic)                          │
│    - Fast (~20-50 cycles)                           │
│    - Medium (few MB)                                │
│    - Hardware managed                               │
└─────────────────────────────────────────────────────┘
                      ↓ 5-10x slower
┌─────────────────────────────────────────────────────┐
│ 4. Global Memory / HBM (Device memory)              │
│    - Slow (~200-400 cycles)                         │
│    - Large (GB-scale)                               │
│    - Accessible by all threads                      │
│    - Main memory for GPU                            │
└─────────────────────────────────────────────────────┘
                      ↓ 100x+ slower
┌─────────────────────────────────────────────────────┐
│ 5. Host Memory (CPU RAM)                            │
│    - Very slow (~10,000+ cycles via PCIe)          │
│    - Large (GB-scale)                               │
│    - Requires explicit transfer                     │
└─────────────────────────────────────────────────────┘
```

### Memory Performance Impact

**Example: Sum 1 million numbers**

```cuda
// BAD: Global memory every time (slow!)
__global__ void sum_global(float* data, float* result) {
    for (int i = 0; i < 1000000; i++) {
        result[0] += data[i];  // Each access: ~400 cycles!
    }
}
// Time: ~400,000,000 cycles = very slow

// GOOD: Use shared memory (fast!)
__global__ void sum_shared(float* data, float* result) {
    __shared__ float temp[256];

    // Each thread loads to shared memory (fast!)
    temp[threadIdx.x] = data[threadIdx.x];
    __syncthreads();

    // Reduce in shared memory (~5 cycles per access)
    // ... reduction logic ...
}
// Time: ~100x faster!
```

---

## Part 4: GPU Architecture Components

### Streaming Multiprocessor (SM)

**The Core Processing Unit**:

```
Streaming Multiprocessor (SM)
┌────────────────────────────────────┐
│ CUDA Cores (128 on Ampere)         │
│ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐        │
│ └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘ ...     │
│                                     │
│ Tensor Cores (4 on Ampere)         │
│ ┌──┐┌──┐┌──┐┌──┐                 │
│ └──┘└──┘└──┘└──┘                 │
│                                     │
│ Shared Memory / L1 Cache (128 KB)  │
│ ┌─────────────────────────┐       │
│ │  Configurable L1/SMEM    │       │
│ └─────────────────────────┘       │
│                                     │
│ Register File (256 KB)             │
│ ┌─────────────────────────┐       │
│ │   65,536 registers       │       │
│ └─────────────────────────┘       │
│                                     │
│ Warp Schedulers (4)                │
│ [Scheduler] [Scheduler] ...        │
└────────────────────────────────────┘
```

**Key Points**:
- Each SM can run **multiple thread blocks** concurrently
- Warp schedulers pick which warp executes each cycle
- Shared memory enables **fast inter-thread communication**

---

### Complete GPU Architecture

**Example: NVIDIA A100**

```
┌────────────────────────────────────────────────────┐
│                    A100 GPU                         │
│                                                     │
│  108 Streaming Multiprocessors (SMs)               │
│  ┌──┐┌──┐┌──┐┌──┐┌──┐                   ┌──┐    │
│  │SM││SM││SM││SM││SM│ ... (108 total)   │SM│    │
│  └──┘└──┘└──┘└──┘└──┘                   └──┘    │
│                                                     │
│  L2 Cache (40 MB)                                  │
│  ┌─────────────────────────────────────┐          │
│  │         Shared L2 Cache              │          │
│  └─────────────────────────────────────┘          │
│                                                     │
│  HBM2 Memory (40 GB, 1555 GB/s bandwidth)         │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐                │
│  │HBM ││HBM ││HBM ││HBM ││HBM │  (5 stacks)     │
│  └────┘└────┘└────┘└────┘└────┘                │
│                                                     │
│  Total: 6,912 CUDA Cores per SM × 108 SMs         │
│       = 746,496 CUDA Cores!                        │
│                                                     │
│  Peak FP32: 19.5 TFLOPS                            │
│  Peak FP16: 312 TFLOPS (with Tensor Cores)        │
└────────────────────────────────────────────────────┘
```

---

## Part 5: Compute Capability

### What is Compute Capability?

**Definition**: GPU's feature set and capabilities, denoted as X.Y (e.g., 8.0, 8.6)

**Major Versions** (First digit):
- **5.x**: Maxwell (GTX 900 series)
- **6.x**: Pascal (GTX 10 series, P100)
- **7.x**: Volta/Turing (V100, RTX 20 series)
- **8.x**: Ampere (A100, RTX 30 series)
- **9.x**: Hopper (H100)

**Why It Matters**:
- Determines available features (Tensor Cores, etc.)
- Performance characteristics differ
- Code optimization targets specific architectures

**Check Your GPU**:
```bash
# Using nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv

# Or in CUDA code
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
```

---

## Part 6: Real-World Performance Analysis

### Case Study: Matrix Multiplication

**Problem**: Multiply two 4096×4096 matrices (FP32)

**Computation**: 4096³ × 2 = 137 billion FLOPs

#### CPU Performance (Intel Core i9-12900K)
```
Peak FP32: ~1.3 TFLOPS (AVX-512)
Time: 137 GFLOPS / 1.3 TFLOPS = 105 ms
Actual (with caching, etc.): ~150 ms
```

#### GPU Performance (RTX 3080)
```
Peak FP32: 29.8 TFLOPS
Time: 137 GFLOPS / 29.8 TFLOPS = 4.6 ms
Actual (optimized cuBLAS): ~6 ms

Speedup: 150ms / 6ms = 25x faster!
```

#### Why Such Speedup?
1. **More compute**: 29.8 vs 1.3 TFLOPS (23x more)
2. **Higher memory bandwidth**: 760 vs 50 GB/s (15x more)
3. **Massive parallelism**: 8,704 cores vs 16 cores (544x more)

---

## Part 7: Identifying GPU-Friendly Problems

### ✅ Good for GPU

**Characteristics**:
- **Data parallel**: Same operation on many data elements
- **Regular access patterns**: Predictable memory access
- **High compute intensity**: Lots of math per memory access
- **Large datasets**: Enough work to saturate GPU

**Examples**:
```python
# Matrix operations
C = A @ B  # Perfect for GPU!

# Image processing
for pixel in image:
    output[pixel] = filter(input[pixel])  # Parallel!

# Deep learning
for neuron in layer:
    activation = relu(weights @ inputs + bias)  # Parallel!

# Physical simulations
for particle in particles:
    update_position(particle)  # Parallel!
```

### ❌ Bad for GPU

**Characteristics**:
- **Sequential dependencies**: Each step needs previous result
- **Irregular patterns**: Random memory access
- **Low compute intensity**: Mostly memory movement
- **Small datasets**: Not enough parallelism
- **Heavy branching**: Different threads take different paths

**Examples**:
```python
# Sequential Fibonacci
for i in range(n):
    fib[i] = fib[i-1] + fib[i-2]  # Sequential dependency!

# Hash table lookup
result = hash_table[random_key()]  # Random access pattern!

# Parsing
while (token = get_next_token()):
    process(token)  # Sequential, unpredictable!
```

---

## Summary

### Key Concepts

**CPU vs GPU**:
- CPU: Few powerful cores, complex control, good at sequential
- GPU: Many simple cores, simple control, great at parallel

**SIMT Execution**:
- 32 threads (warp) execute same instruction together
- Branch divergence causes serialization
- Design algorithms to minimize divergence

**Memory Hierarchy**:
- Registers (fastest) → Shared Memory → L1/L2 Cache → Global Memory → Host Memory (slowest)
- Optimize by using faster memory levels

**GPU Architecture**:
- Streaming Multiprocessors (SMs) are core processing units
- Each SM has CUDA cores, shared memory, registers
- Multiple SMs work in parallel

**When to Use GPU**:
- Data parallel problems
- Regular memory patterns
- High compute intensity
- Large datasets

---

## Practice Questions

### Question 1
**Q**: You need to process a 4K image (8.3M pixels) by applying the same filter to each pixel. Should you use CPU or GPU? Why?

<details>
<summary>Answer</summary>

**GPU**!

Reasons:
- Data parallel: Same filter applied to all pixels
- Regular pattern: Process each pixel independently
- Large dataset: 8.3M pixels = lots of parallelism
- High compute: Filter involves multiple operations per pixel

Expected speedup: 20-100x depending on filter complexity
</details>

### Question 2
**Q**: Why does branch divergence hurt GPU performance?

<details>
<summary>Answer</summary>

Because all 32 threads in a warp must execute the same instruction.

When branches diverge:
1. Execute first path, threads not taking it are IDLE
2. Execute second path, threads not taking it are IDLE
3. Result: Serialization instead of parallelism!

Example: If 16 threads take if-branch and 16 take else-branch, you lose 50% performance.
</details>

### Question 3
**Q**: Rank these memory types from fastest to slowest: Global Memory, Registers, Shared Memory, L2 Cache

<details>
<summary>Answer</summary>

Fastest to Slowest:
1. **Registers** (~1 cycle)
2. **Shared Memory** (~5-20 cycles)
3. **L2 Cache** (~200 cycles)
4. **Global Memory** (~400 cycles)

Performance ratio: Registers are ~400x faster than Global Memory!
</details>

---

## Next Steps

Now that you understand GPU architecture:

1. ✅ **Next Tutorial**: [CUDA Programming Model](./cuda-programming-model.md)
   - Learn about grids, blocks, threads in detail
   - Understand kernel launch syntax

2. ✅ **Setup**: [Development Environment](./environment-setup.md)
   - Install CUDA Toolkit
   - Configure your IDE

3. ✅ **Start Coding**: [Your First CUDA Kernel](./your-first-cuda-kernel.md)
   - Write and run your first kernel
   - See the concepts in action!

---

**Tutorial Generated by**: Content Creator Agent #3
**Session**: session_20251119_051648
**Difficulty**: Foundation (L0)
**Time**: 45-60 minutes
**Next**: [CUDA Programming Model Deep Dive](./cuda-programming-model.md)
