# Project: Custom Activation Function Library

## Overview

**Level**: L1 (Beginner)
**Estimated Time**: 6-8 hours
**Difficulty**: ⭐⭐ (Beginner)
**Type**: Implementation

Build a complete CUDA-accelerated activation function library with PyTorch integration, testing, and benchmarking.

---

## Learning Objectives

By completing this project, you will:
- ✅ Implement multiple CUDA kernels from scratch
- ✅ Create PyTorch custom operations
- ✅ Write comprehensive tests
- ✅ Benchmark GPU performance
- ✅ Compare with PyTorch native implementations
- ✅ Package reusable CUDA code

---

## Prerequisites

**Required Knowledge**:
- L1: Thread hierarchy (See: [tutorials/beginner/thread-hierarchy.md](../../../tutorials/beginner/thread-hierarchy.md))
- L1: Element-wise operations (See: [tutorials/beginner/element-wise-ops.md](../../../tutorials/beginner/element-wise-ops.md))
- Basic PyTorch usage

**Required Skills**:
- Write simple CUDA kernels
- Compile CUDA code
- Basic Python programming

**Software Requirements**:
- CUDA >= 11.0
- PyTorch >= 2.0
- Python >= 3.8
- pytest (for testing)

---

## Problem Statement

### Background

Activation functions are critical components of neural networks. While PyTorch provides optimized implementations, building your own teaches fundamental GPU programming concepts and is a realistic first project.

### The Challenge

Create a **production-quality activation function library** that:
1. Implements 6+ activation functions
2. Provides both forward and backward passes
3. Integrates seamlessly with PyTorch
4. Matches or exceeds PyTorch's performance
5. Includes comprehensive tests

---

## Functional Requirements

### Must Have (Core Functionality)

#### 1. Activation Functions

Implement these activation functions:

**ReLU (Rectified Linear Unit)**:
```
forward:  f(x) = max(0, x)
backward: f'(x) = 1 if x > 0 else 0
```

**LeakyReLU**:
```
forward:  f(x) = x if x > 0 else alpha * x
backward: f'(x) = 1 if x > 0 else alpha
```

**GELU (Gaussian Error Linear Unit)**:
```
forward:  f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
backward: [derivative using chain rule]
```

**SiLU / Swish**:
```
forward:  f(x) = x * sigmoid(x)
backward: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

**Mish**:
```
forward:  f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
backward: [derivative using chain rule]
```

**ELU (Exponential Linear Unit)**:
```
forward:  f(x) = x if x > 0 else alpha * (e^x - 1)
backward: f'(x) = 1 if x > 0 else f(x) + alpha
```

#### 2. PyTorch Integration

```python
import torch
from activation_lib import relu, gelu, swish

# Should work like PyTorch functions
x = torch.randn(1024, 1024, device='cuda')
y = relu(x)  # Uses your CUDA kernel
y.backward(torch.ones_like(y))  # Autograd integration
```

#### 3. Testing

- Correctness tests (compare with PyTorch)
- Gradient tests (using PyTorch's gradcheck)
- Edge case tests (zeros, infinities, NaNs)
- Different tensor shapes

#### 4. Benchmarking

- Performance vs PyTorch
- Scaling with tensor size
- Throughput measurements (GB/s)

### Should Have (Important Features)

- **In-place operations**: `relu_` modifies tensor in-place (saves memory)
- **Fused operations**: `gelu_backward` combines backward computation
- **Multiple data types**: FP32, FP16 support

### Could Have (Extensions)

- Additional activations (PReLU, Hardswish, etc.)
- Batched operations
- 2D/3D specializations for images
- Auto-tuning block sizes

---

## Technical Specifications

### Project Structure

```
activation-library/
├── src/
│   ├── relu.cu              # ReLU kernels
│   ├── gelu.cu              # GELU kernels
│   ├── swish.cu             # Swish kernels
│   ├── mish.cu              # Mish kernels
│   ├── elu.cu               # ELU kernels
│   ├── leaky_relu.cu        # LeakyReLU kernels
│   └── utils.h              # Common utilities
│
├── python/
│   ├── __init__.py          # Package initialization
│   ├── relu.py              # ReLU PyTorch binding
│   ├── gelu.py              # GELU PyTorch binding
│   ├── swish.py             # Swish PyTorch binding
│   ├── mish.py              # Mish PyTorch binding
│   ├── elu.py               # ELU PyTorch binding
│   └── leaky_relu.py        # LeakyReLU PyTorch binding
│
├── tests/
│   ├── test_correctness.py  # Correctness tests
│   ├── test_gradients.py    # Gradient tests
│   ├── test_edge_cases.py   # Edge case tests
│   └── test_shapes.py       # Different tensor shapes
│
├── benchmarks/
│   ├── benchmark_forward.py     # Forward pass benchmarks
│   ├── benchmark_backward.py    # Backward pass benchmarks
│   └── benchmark_all.py         # Complete benchmarks
│
├── examples/
│   └── simple_usage.py      # Usage examples
│
├── setup.py                 # Installation script
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## Implementation Roadmap

### Phase 1: Foundation (2 hours)

**Goal**: Get one activation (ReLU) working end-to-end

**Tasks**:
1. [ ] Set up project structure
2. [ ] Implement ReLU forward CUDA kernel
3. [ ] Implement ReLU backward CUDA kernel
4. [ ] Create Python bindings using `torch.utils.cpp_extension.load`
5. [ ] Write basic test

**Validation**:
- [ ] ReLU kernel compiles
- [ ] Forward pass matches PyTorch
- [ ] Backward pass matches PyTorch
- [ ] Test passes

**Example ReLU Kernel**:
```cuda
// src/relu.cu
__global__ void relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}
```

---

### Phase 2: More Activations (2-3 hours)

**Goal**: Implement remaining activation functions

**Tasks**:
1. [ ] Implement LeakyReLU
2. [ ] Implement GELU (use approximation)
3. [ ] Implement Swish/SiLU
4. [ ] Implement Mish
5. [ ] Implement ELU

**Validation**:
- [ ] Each activation compiles
- [ ] Forward matches PyTorch (tolerance: 1e-4)
- [ ] Backward matches PyTorch (tolerance: 1e-3)

**Tips**:
- Use `tanhf()`, `expf()` for math functions
- Be careful with numerical stability
- GELU approximation is fine (exact is expensive)

---

### Phase 3: PyTorch Integration (1-2 hours)

**Goal**: Clean Python API with autograd support

**Tasks**:
1. [ ] Create `torch.autograd.Function` for each activation
2. [ ] Implement forward() and backward() methods
3. [ ] Add shape checking and error handling
4. [ ] Create convenient function wrappers
5. [ ] Add docstrings

**Example PyTorch Binding**:
```python
# python/relu.py
import torch
from torch.utils.cpp_extension import load

# Load CUDA extension
relu_cuda = load(
    name='relu_cuda',
    sources=['src/relu.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.empty_like(input)
        relu_cuda.forward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.empty_like(input)
        relu_cuda.backward(grad_output, input, grad_input)
        return grad_input

def relu(input):
    """Apply ReLU activation function.

    Args:
        input: Input tensor (must be CUDA tensor)

    Returns:
        Output tensor with ReLU applied
    """
    assert input.is_cuda, "Input must be a CUDA tensor"
    return ReLUFunction.apply(input)
```

---

### Phase 4: Testing (1-2 hours)

**Goal**: Comprehensive test coverage

**Tasks**:
1. [ ] Write correctness tests for all activations
2. [ ] Add gradient checking tests
3. [ ] Test edge cases (0, inf, -inf, nan)
4. [ ] Test different shapes (1D, 2D, 3D, etc.)
5. [ ] Test different data types (FP32, FP16)

**Example Test**:
```python
# tests/test_correctness.py
import torch
import pytest
from activation_lib import relu

def test_relu_forward():
    """Test ReLU forward pass."""
    x = torch.randn(1000, device='cuda')

    # Custom implementation
    y_custom = relu(x)

    # PyTorch reference
    y_torch = torch.relu(x)

    # Compare
    torch.testing.assert_close(y_custom, y_torch, rtol=1e-5, atol=1e-5)

def test_relu_backward():
    """Test ReLU backward pass."""
    x = torch.randn(1000, device='cuda', requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    # Forward
    y = relu(x)
    y_ref = torch.relu(x_ref)

    # Backward
    y.sum().backward()
    y_ref.sum().backward()

    # Compare gradients
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("size", [10, 100, 1000, 10000, 100000])
def test_relu_sizes(size):
    """Test different tensor sizes."""
    x = torch.randn(size, device='cuda')
    y = relu(x)
    assert y.shape == x.shape
```

**Run tests**:
```bash
pytest tests/ -v
```

---

### Phase 5: Benchmarking (1 hour)

**Goal**: Measure performance and compare with PyTorch

**Tasks**:
1. [ ] Benchmark forward pass for all activations
2. [ ] Benchmark backward pass
3. [ ] Test different tensor sizes
4. [ ] Generate performance report

**Example Benchmark**:
```python
# benchmarks/benchmark_forward.py
import torch
import time
from activation_lib import relu

def benchmark_forward(activation_fn, name, sizes, iterations=100):
    """Benchmark forward pass."""
    print(f"\n{name} Forward Pass Benchmark")
    print(f"{'Size':<15} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}")
    print("-" * 50)

    for size in sizes:
        x = torch.randn(size, device='cuda')

        # Warmup
        for _ in range(10):
            _ = activation_fn(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(iterations):
            y = activation_fn(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / iterations * 1000  # ms

        # Calculate bandwidth (read input + write output)
        bytes_accessed = 2 * size * 4  # float32
        bandwidth = (bytes_accessed / (elapsed * 1e-3)) / 1e9  # GB/s

        print(f"{size:<15,} {elapsed:<12.4f} {bandwidth:<15.2f}")

if __name__ == "__main__":
    sizes = [1000, 10000, 100000, 1000000, 10000000]

    print("=== Custom ReLU ===")
    benchmark_forward(relu, "ReLU", sizes)

    print("\n=== PyTorch ReLU ===")
    benchmark_forward(torch.relu, "ReLU (PyTorch)", sizes)
```

---

## Performance Targets

### Bandwidth Targets

Activation functions are **memory-bound** (simple compute, lots of data movement).

**Target**: Achieve >80% of memory bandwidth

**Example** (RTX 3080 with 760 GB/s bandwidth):
- Target: >600 GB/s
- Typical achieve: 500-650 GB/s

### Comparison with PyTorch

Your implementation should:
- Match PyTorch performance (within 10%)
- For some activations, may be faster (less overhead)

---

## Grading Rubric

### Excellent (90-100%)
- ✅ All 6 activations implemented correctly
- ✅ All tests pass (correctness, gradients, edge cases)
- ✅ Performance within 10% of PyTorch
- ✅ Clean, well-documented code
- ✅ Comprehensive benchmarks
- ✅ Bonus features implemented

### Good (75-89%)
- ✅ All 6 activations implemented
- ✅ Most tests pass (>90%)
- ✅ Performance within 20% of PyTorch
- ✅ Code is functional and documented
- ✅ Basic benchmarks provided

### Satisfactory (60-74%)
- ✅ At least 4 activations implemented
- ✅ Basic tests pass
- ✅ Code compiles and runs
- ✅ Some documentation

### Needs Improvement (<60%)
- ❌ Fewer than 4 activations
- ❌ Tests failing
- ❌ Significant correctness issues

---

## Extensions & Challenges

### Extension 1: In-Place Operations (Medium)

Implement in-place versions:
```python
def relu_(input):
    """In-place ReLU (modifies input tensor)."""
    # Modify input directly, save memory
```

### Extension 2: Fused Backward (Medium)

Combine backward computation with gradient scaling:
```cuda
// Fused: grad_input = grad_output * relu'(input) * scale
__global__ void relu_backward_fused(grad_output, input, grad_input, scale, N);
```

### Extension 3: FP16 Support (Hard)

Support half-precision:
```python
x_fp16 = torch.randn(1000, device='cuda', dtype=torch.float16)
y = relu(x_fp16)  # Should work!
```

### Extension 4: Adaptive Block Size (Hard)

Auto-tune block size based on tensor size:
```cpp
int choose_block_size(int N) {
    if (N < 1024) return 128;
    else if (N < 1024*1024) return 256;
    else return 512;
}
```

### Challenge: Beat PyTorch

Can you make your implementation faster than PyTorch for any activation?

**Hint**: Try fusing operations or optimizing for specific tensor sizes.

---

## Resources

### Recommended Reading
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Activation Functions Explained](https://arxiv.org/abs/1710.05941)

### Reference Implementations
- PyTorch: [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- LeetCUDA examples: `kernels/relu/`, `kernels/gelu/`, `kernels/swish/`

### Debugging Tips
- **Compilation errors**: Check CUDA syntax, semicolons, template brackets
- **Wrong results**: Print intermediate values, compare with PyTorch
- **Slow performance**: Profile with NSight, check memory access patterns
- **Gradient errors**: Use `torch.autograd.gradcheck()`

---

## Submission

### What to Submit
1. Complete source code (src/ directory)
2. Python bindings (python/ directory)
3. All tests passing (tests/ directory)
4. Benchmark results (benchmarks/)
5. Brief report (see template below)

### Report Template

```markdown
# Activation Library - Project Report

## Implementation Summary
- Activations implemented: [list]
- Total lines of CUDA code: [number]
- Development time: [hours]

## Key Decisions
- Block size chosen: [value] (rationale: ...)
- Math functions used: [exp, tanh, etc.]
- Optimization techniques: [list]

## Challenges Faced
1. [Challenge]: [Solution]
2. [Challenge]: [Solution]

## Performance Results
[Table showing your performance vs PyTorch]

| Activation | Size    | Your Time (ms) | PyTorch (ms) | Speedup |
|------------|---------|----------------|--------------|---------|
| ReLU       | 1M      | 0.05           | 0.05         | 1.0x    |
| GELU       | 1M      | 0.12           | 0.13         | 1.08x   |
...

## Learnings
- [Key learning 1]
- [Key learning 2]
- [Key learning 3]
```

---

## Getting Started

```bash
# 1. Create project directory
mkdir -p activation-library/{src,python,tests,benchmarks,examples}
cd activation-library

# 2. Copy starter code (if provided)
# Or start from scratch!

# 3. Implement ReLU first
nvim src/relu.cu

# 4. Test as you go
python -c "from python.relu import relu; import torch; print(relu(torch.randn(100, device='cuda')))"

# 5. Add tests
pytest tests/test_correctness.py -v

# 6. Benchmark
python benchmarks/benchmark_forward.py
```

---

**Good luck! Remember: Start simple (ReLU), test early, test often!**

---

**Project Designed by**: Project Designer Agent #5
**Session**: session_20251119_051648
**Difficulty**: Beginner
**Related Tutorials**: [Element-wise Ops](../../../tutorials/beginner/element-wise-ops.md)
