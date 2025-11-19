# Tutorial: Element-wise Operations in CUDA

## Learning Objectives

By the end of this tutorial, you will:
- ✅ Implement activation functions (ReLU, GELU, Swish) in CUDA
- ✅ Create PyTorch custom CUDA extensions
- ✅ Benchmark kernel performance against PyTorch
- ✅ Understand bandwidth-bound vs compute-bound operations
- ✅ Write fused forward-backward kernels

**Level**: L1 (Beginner)
**Time**: 75-90 minutes
**Prerequisites**: Your First CUDA Kernel, Memory Model Basics

---

## Introduction

### What are Element-wise Operations?

**Element-wise operations** process each element independently:

```python
# Element-wise: each output depends on ONE input element
y[i] = f(x[i])

# Examples:
y[i] = relu(x[i]) = max(0, x[i])
y[i] = x[i] * 2.0
y[i] = exp(x[i])
```

**Contrast with reductions** (each output depends on MANY inputs):
```python
y = sum(x)  # One output from all inputs
```

### Why Start With Element-wise?

1. **Simplest pattern** - perfect for learning CUDA
2. **Common in ML** - activation functions, normalization, dropout
3. **Easy to optimize** - straightforward memory access
4. **Great for benchmarking** - clear performance metrics

### Activation Functions Primer

Activation functions introduce non-linearity in neural networks:

| Function | Formula | Used In |
|----------|---------|---------|
| **ReLU** | `max(0, x)` | Most models (default choice) |
| **GELU** | `x * Φ(x)` (Gaussian) | Transformers (BERT, GPT) |
| **Swish/SiLU** | `x * sigmoid(x)` | Modern CNNs, EfficientNet |
| **Tanh** | `(e^x - e^-x)/(e^x + e^-x)` | RNNs (older models) |

---

## Part 1: Implementing ReLU

### The Simplest Activation

**ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`

**Forward pass**:
```
f(x) = max(0, x) = { x  if x > 0
                   { 0  if x ≤ 0
```

**Backward pass** (derivative):
```
f'(x) = { 1  if x > 0
        { 0  if x ≤ 0
```

### CUDA Implementation

```cuda
// relu.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Forward kernel
__global__ void relu_forward(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        y[i] = fmaxf(0.0f, x[i]);  // fmaxf is GPU-optimized max for floats
    }
}

// Backward kernel
__global__ void relu_backward(const float* x, const float* grad_y, float* grad_x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        grad_x[i] = (x[i] > 0.0f) ? grad_y[i] : 0.0f;
    }
}

// Host wrapper
void relu_cuda(const float* x, float* y, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    relu_forward<<<blocks, threads>>>(x, y, N);
    cudaDeviceSynchronize();
}
```

**Key points**:
- `fmaxf()` is optimized for GPU (faster than `max()` or ternary operator)
- Backward pass only needs input `x` (gradient doesn't depend on `y`)
- Very simple - each thread does ONE operation

### Complete Example with Timing

```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void relu_forward(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

int main() {
    int N = 1 << 24;  // 16M elements
    size_t bytes = N * sizeof(float);

    // Allocate
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    // Initialize: random values in [-1, 1]
    for (int i = 0; i < N; i++) {
        h_x[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warm-up
    relu_forward<<<blocks, threads>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    relu_forward<<<blocks, threads>>>(d_x, d_y, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Bandwidth calculation
    // Read N floats, write N floats = 2N * 4 bytes
    float bandwidth_GB = (2.0f * bytes) / (ms * 1e6);

    printf("ReLU performance:\n");
    printf("  Time: %.3f ms\n", ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_GB);
    printf("  Throughput: %.2f billion elements/s\n", N / (ms * 1e6));

    // Verify correctness
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < 100; i++) {  // Check first 100
        float expected = (h_x[i] > 0.0f) ? h_x[i] : 0.0f;
        if (fabsf(h_y[i] - expected) > 1e-5f) {
            printf("Error at %d: expected %.3f, got %.3f\n", i, expected, h_y[i]);
            correct = false;
            break;
        }
    }
    if (correct) printf("✓ Correctness verified!\n");

    // Cleanup
    free(h_x); free(h_y);
    cudaFree(d_x); cudaFree(d_y);

    return 0;
}
```

**Compile and run**:
```bash
nvcc relu.cu -o relu
./relu
```

**Expected output** (RTX 3080):
```
ReLU performance:
  Time: 0.145 ms
  Bandwidth: 550 GB/s
  Throughput: 110 billion elements/s
✓ Correctness verified!
```

**Analysis**: Bandwidth is ~70% of peak (760 GB/s) - excellent for such a simple kernel!

---

## Part 2: GELU - A More Complex Activation

### What is GELU?

**GELU (Gaussian Error Linear Unit)**: Smooth approximation of ReLU

**Exact formula**:
```
GELU(x) = x * Φ(x)
where Φ(x) = CDF of standard normal distribution
           = 0.5 * (1 + erf(x / √2))
```

**Fast approximation** (used in practice):
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Why GELU?**
- Used in BERT, GPT, ViT (most Transformers!)
- Smooth (differentiable everywhere)
- Better gradient flow than ReLU

### CUDA Implementation

```cuda
__global__ void gelu_forward(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float xi = x[i];

        // Fast approximation
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
        const float coef = 0.044715f;

        float x_cubed = xi * xi * xi;
        float inner = sqrt_2_over_pi * (xi + coef * x_cubed);
        float tanh_inner = tanhf(inner);

        y[i] = 0.5f * xi * (1.0f + tanh_inner);
    }
}

__global__ void gelu_backward(const float* x, const float* grad_y, float* grad_x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float xi = x[i];

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coef = 0.044715f;

        float x_squared = xi * xi;
        float x_cubed = x_squared * xi;

        float inner = sqrt_2_over_pi * (xi + coef * x_cubed);
        float tanh_inner = tanhf(inner);
        float sech_squared = 1.0f - tanh_inner * tanh_inner;  // sech²(x) = 1 - tanh²(x)

        float d_inner_dx = sqrt_2_over_pi * (1.0f + 3.0f * coef * x_squared);

        // Chain rule
        float gelu_grad = 0.5f * (1.0f + tanh_inner) +
                         0.5f * xi * sech_squared * d_inner_dx;

        grad_x[i] = grad_y[i] * gelu_grad;
    }
}
```

**Key differences from ReLU**:
- Many more operations (multiply, tanh, etc.)
- Still element-wise (no thread cooperation needed)
- Backward pass is more complex

### Performance Comparison

```cuda
int main() {
    int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    // ... allocation code ...

    // Benchmark ReLU
    cudaEventRecord(start);
    relu_forward<<<blocks, threads>>>(d_x, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_relu;
    cudaEventElapsedTime(&ms_relu, start, stop);

    // Benchmark GELU
    cudaEventRecord(start);
    gelu_forward<<<blocks, threads>>>(d_x, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_gelu;
    cudaEventElapsedTime(&ms_gelu, start, stop);

    printf("ReLU:  %.3f ms (%.2f GB/s)\n", ms_relu, 2*bytes/(ms_relu*1e6));
    printf("GELU:  %.3f ms (%.2f GB/s)\n", ms_gelu, 2*bytes/(ms_gelu*1e6));
    printf("GELU is %.2fx slower than ReLU\n", ms_gelu / ms_relu);

    // ... cleanup ...
}
```

**Expected results** (RTX 3080):
```
ReLU:  0.145 ms (550 GB/s)
GELU:  0.310 ms (257 GB/s)
GELU is 2.14x slower than ReLU
```

**Why slower?**
- ReLU: Memory-bound (limited by bandwidth)
- GELU: Partially compute-bound (more math per element)
- GELU has lower bandwidth utilization but more FLOPs

---

## Part 3: PyTorch Integration

### Creating a Custom CUDA Extension

Let's integrate our kernels into PyTorch!

**File structure**:
```
relu_cuda/
├── relu_kernel.cu      # CUDA kernels
├── relu_cuda.cpp       # C++ wrapper
└── setup.py            # Build script
```

### Step 1: CUDA Kernel File

`relu_kernel.cu`:
```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

__global__ void relu_backward_kernel(const float* x, const float* grad_y, float* grad_x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        grad_x[i] = (x[i] > 0.0f) ? grad_y[i] : 0.0f;
    }
}

// C++ wrapper for CUDA kernels
torch::Tensor relu_forward_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int N = x.numel();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N
    );

    return y;
}

torch::Tensor relu_backward_cuda(torch::Tensor x, torch::Tensor grad_y) {
    auto grad_x = torch::empty_like(x);
    int N = x.numel();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        N
    );

    return grad_x;
}
```

### Step 2: C++ Interface

`relu_cuda.cpp`:
```cpp
#include <torch/extension.h>

// Declare CUDA functions
torch::Tensor relu_forward_cuda(torch::Tensor x);
torch::Tensor relu_backward_cuda(torch::Tensor x, torch::Tensor grad_y);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &relu_forward_cuda, "ReLU forward (CUDA)");
    m.def("backward", &relu_backward_cuda, "ReLU backward (CUDA)");
}
```

### Step 3: Build Script

`setup.py`:
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='relu_cuda',
    ext_modules=[
        CUDAExtension('relu_cuda', [
            'relu_cuda.cpp',
            'relu_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

### Step 4: Build and Install

```bash
python setup.py install
```

### Step 5: Use in PyTorch

```python
import torch
import relu_cuda

# Create test tensor
x = torch.randn(1024, 1024, device='cuda')

# Forward pass
y = relu_cuda.forward(x)

# Backward pass (with gradient from next layer)
grad_y = torch.ones_like(y)
grad_x = relu_cuda.backward(x, grad_y)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Gradient shape: {grad_x.shape}")

# Verify against PyTorch
y_torch = torch.relu(x)
print(f"Max difference: {(y - y_torch).abs().max().item()}")
```

### Step 6: Autograd Integration

For automatic differentiation, wrap in a `torch.autograd.Function`:

```python
import torch
import relu_cuda

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return relu_cuda.forward(x)

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        return relu_cuda.backward(x, grad_y)

# Easy-to-use wrapper
def relu(x):
    return ReLUFunction.apply(x)

# Now it works with autograd!
x = torch.randn(1000, 1000, device='cuda', requires_grad=True)
y = relu(x)
loss = y.sum()
loss.backward()

print(f"Gradient computed: {x.grad.shape}")
```

---

## Part 4: Benchmarking Against PyTorch

### Comprehensive Benchmark

```python
import torch
import time
import relu_cuda

def benchmark(func, x, n_iters=1000, warmup=100):
    # Warm-up
    for _ in range(warmup):
        func(x)
    torch.cuda.synchronize()

    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iters):
        func(x)
    end.record()

    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n_iters

    return ms

# Test sizes
sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

print("Size\t\tPyTorch\t\tCustom\t\tSpeedup")
print("-" * 60)

for N in sizes:
    x = torch.randn(N, N, device='cuda')

    ms_torch = benchmark(lambda x: torch.relu(x), x)
    ms_custom = benchmark(lambda x: relu_cuda.forward(x), x)

    speedup = ms_torch / ms_custom

    print(f"{N}x{N}\t\t{ms_torch:.3f} ms\t{ms_custom:.3f} ms\t{speedup:.2f}x")
```

**Expected results**:
```
Size        PyTorch     Custom      Speedup
------------------------------------------------------------
1024x1024   0.012 ms    0.010 ms    1.20x
4096x4096   0.145 ms    0.140 ms    1.04x
16384x16384 2.310 ms    2.280 ms    1.01x
```

**Observations**:
- Custom kernel is slightly faster (less overhead)
- For simple operations, PyTorch is already very optimized
- Big wins come from **fused operations** (next section!)

---

## Part 5: Kernel Fusion

### The Problem: Multiple Kernel Launches

```python
# Typical deep learning: many element-wise ops
x = input
x = layer_norm(x)      # Kernel 1
x = linear(x, W)       # Kernel 2 (GEMM)
x = gelu(x)            # Kernel 3
x = dropout(x)         # Kernel 4
```

**Issue**: Each kernel launch has overhead + separate memory reads/writes

### The Solution: Fused Kernels

Combine multiple operations in ONE kernel:

```cuda
// Fused: GELU + Dropout
__global__ void gelu_dropout_fused(
    const float* x,
    float* y,
    float* mask,  // Dropout mask
    float p_keep,
    unsigned long long seed,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float xi = x[i];

        // GELU computation
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coef = 0.044715f;
        float x_cubed = xi * xi * xi;
        float inner = sqrt_2_over_pi * (xi + coef * x_cubed);
        float gelu_out = 0.5f * xi * (1.0f + tanhf(inner));

        // Dropout
        // Simple random (use curand for production!)
        unsigned long long state = seed + i;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float rand_val = (float)(state >> 32) / 4294967296.0f;

        float keep = (rand_val < p_keep) ? 1.0f : 0.0f;
        mask[i] = keep;

        // Combined output
        y[i] = (gelu_out * keep) / p_keep;  // Scale by 1/p_keep
    }
}
```

**Benefits**:
- Read `x` once (instead of twice)
- Write `y` once (instead of intermediate buffer)
- One kernel launch (less overhead)
- **2-3x faster** than separate kernels!

---

## Part 6: Performance Analysis

### Memory Bandwidth Analysis

For element-wise operations:

**Arithmetic Intensity** = FLOPs / Bytes Transferred

**ReLU**:
- Operations: 1 comparison
- Memory: Read 4 bytes, write 4 bytes = 8 bytes
- Intensity: ~0.1 FLOP/byte (VERY low)

**GELU**:
- Operations: ~15 FLOPs (multiply, add, tanh, etc.)
- Memory: 8 bytes
- Intensity: ~2 FLOP/byte (still low)

**Classification**:
- Intensity < 10: **Memory-bound**
- Intensity > 100: **Compute-bound**

Most activations are **memory-bound** → optimize memory access!

### Roofline Model

```
Performance (GFLOPS)
     │
     │     Compute Bound
     │    ╱
     │   ╱
     │  ╱_____________ Roofline (Peak Compute)
     │ ╱│
     │╱ │ Memory Bound
     ├──┴──────────────────────────── Arithmetic Intensity
     0  1  10  100  1000
```

**ReLU and most activations**: Stuck in memory-bound region!

**Solution**: Fuse with compute-heavy ops (GEMM, convolution)

---

## Exercises

### Exercise 1: Implement Swish/SiLU

Swish activation: `f(x) = x * sigmoid(x) = x / (1 + exp(-x))`

**Tasks**:
1. Implement forward and backward kernels
2. Benchmark against PyTorch's `torch.nn.SiLU`
3. Create PyTorch extension

**Hints**:
- Use `1.0f / (1.0f + expf(-x))` for sigmoid
- Backward: `f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`

### Exercise 2: Fused ReLU + Scale

Implement: `y = max(0, x) * alpha` in a single kernel

Compare performance to:
```python
y = torch.relu(x) * alpha  # Two operations
```

### Exercise 3: Vectorized Element-wise

Modify ReLU to process 4 floats at once using `float4`:

```cuda
__global__ void relu_vectorized(const float* x, float* y, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (i + 3 < N) {
        float4 val = *((float4*)&x[i]);
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        *((float4*)&y[i]) = val;
    }
}
```

**Question**: Does this improve performance? Why or why not?

### Solutions

<details>
<summary>Exercise 1: Swish Solution</summary>

```cuda
__global__ void swish_forward(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = x[i];
        float sigmoid_x = 1.0f / (1.0f + expf(-xi));
        y[i] = xi * sigmoid_x;
    }
}

__global__ void swish_backward(const float* x, const float* grad_y, float* grad_x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = x[i];
        float sigmoid_x = 1.0f / (1.0f + expf(-xi));
        float swish_grad = sigmoid_x + xi * sigmoid_x * (1.0f - sigmoid_x);
        grad_x[i] = grad_y[i] * swish_grad;
    }
}
```
</details>

---

## Key Takeaways

### Element-wise Pattern

```
✅ Each thread processes ONE element
✅ No thread cooperation needed
✅ No synchronization required
✅ Excellent for GPU parallelism
```

### Performance Characteristics

```
Memory-bound: ReLU, Sigmoid, Tanh (simple ops)
  → Optimize memory access patterns
  → Fuse with other operations

Partially compute-bound: GELU, Swish (complex ops)
  → Still benefits from fusion
  → Use fast math approximations
```

### PyTorch Integration

```
1. Write CUDA kernels (.cu)
2. Create C++ wrapper (.cpp)
3. Build with torch.utils.cpp_extension
4. Wrap in torch.autograd.Function
5. Use like native PyTorch!
```

---

## Next Steps

Now that you've mastered element-wise operations:

1. ✅ **[Project: Activation Library](../../projects/beginner/activation-library/)** - Build complete library
2. ✅ **[Reductions Tutorial](./reductions.md)** - Sum, max, mean operations
3. ✅ **[Kernel Fusion Advanced](../advanced/kernel-fusion.md)** - Complex fusion patterns
4. ✅ **[Memory Coalescing](../intermediate/memory-coalescing.md)** - Optimize memory access

---

**Tutorial Generated by**: Content Creator Agent #6
**Session**: session_20251119_051648
**Difficulty**: Beginner
**Next Tutorial**: [Simple Reductions](./simple-reductions.md)
