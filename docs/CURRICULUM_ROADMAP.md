# LeetCUDA Curriculum Roadmap

## Overview

This curriculum roadmap provides a structured learning path from complete beginner to expert GPU programmer. It follows a progressive difficulty model (L0-L5) with clear prerequisites, learning objectives, and estimated timeframes.

---

## Learning Philosophy

### Progressive Complexity
- Start with fundamentals, build to advanced
- Each level builds on previous knowledge
- Clear progression milestones
- No prerequisite gaps

### Hands-On Learning
- Theory + Practice for every topic
- Working code examples
- Real-world applications
- Performance-driven exercises

### Multiple Learning Paths
- **Fast Track**: For experienced programmers (6-8 months)
- **Standard Track**: Comprehensive learning (12-18 months)
- **Deep Dive Track**: Research-oriented (18-24 months)

---

## Difficulty Levels

| Level | Title | Duration | Audience | Focus |
|-------|-------|----------|----------|-------|
| **L0** | Foundation | 1-2 weeks | Complete beginners | GPU basics, setup, PyTorch |
| **L1** | Basic Kernels | 2-3 weeks | CUDA beginners | Simple kernels, thread model |
| **L2** | Memory Optimization | 3-4 weeks | Intermediate | Memory hierarchy, optimization |
| **L3** | Advanced Patterns | 4-6 weeks | Advanced | GEMM, attention, complex algorithms |
| **L4** | Tensor Cores | 6-8 weeks | Expert | Tensor cores, PTX, CUTLASS |
| **L5** | Production Systems | 8-12 weeks | Master | Multi-GPU, production deployment |

---

# Level 0: Foundation

## Overview
**Duration**: 1-2 weeks
**Prerequisite**: Basic programming knowledge (Python or C++)
**Goal**: Understand GPU architecture and set up development environment

## Topics

### 0.1 GPU Architecture Basics
**Learning Time**: 2-3 hours

**Concepts**:
- CPU vs GPU: When and why to use GPUs
- SIMT (Single Instruction, Multiple Thread) model
- Streaming Multiprocessors (SMs)
- Memory hierarchy overview
- Compute capability

**Materials**:
- Reading: GPU architecture fundamentals
- Videos: Visual explanations of GPU design
- Quizzes: Architecture concepts

**Outcomes**:
- [ ] Explain difference between CPU and GPU
- [ ] Understand parallel processing model
- [ ] Know GPU memory hierarchy
- [ ] Identify appropriate GPU tasks

---

### 0.2 CUDA Programming Model
**Learning Time**: 3-4 hours

**Concepts**:
- Kernels, grids, blocks, threads
- Thread hierarchy
- __global__, __device__, __host__ functions
- Basic CUDA syntax
- Compilation process (nvcc)

**Materials**:
- Tutorial: "Your First CUDA Program"
- Examples: Vector addition walkthrough
- Exercises: Modify simple kernels

**Outcomes**:
- [ ] Write basic CUDA kernel
- [ ] Understand thread indexing
- [ ] Compile and run CUDA programs
- [ ] Debug simple CUDA errors

---

### 0.3 Development Environment Setup
**Learning Time**: 2-3 hours

**Components**:
- CUDA Toolkit installation
- Compiler setup (nvcc, gcc/g++)
- IDE configuration (VSCode, CLion)
- PyTorch with CUDA
- Profiling tools (NSight)

**Materials**:
- Step-by-step installation guides (Linux, Windows, macOS)
- Troubleshooting common issues
- Environment verification scripts

**Outcomes**:
- [ ] Install CUDA Toolkit
- [ ] Compile sample CUDA program
- [ ] Run PyTorch with GPU
- [ ] Use basic profiling tools

---

### 0.4 C++ Essentials for CUDA
**Learning Time**: 4-6 hours (skip if proficient)

**Concepts**:
- Pointers and references
- Memory management (new/delete, malloc/free)
- Templates basics
- Structs and classes
- Compilation and linking

**Materials**:
- C++ refresher tutorial
- CUDA-specific C++ patterns
- Common pitfalls

**Outcomes**:
- [ ] Work with pointers confidently
- [ ] Manage memory allocation
- [ ] Use basic templates
- [ ] Understand CUDA C++ extensions

---

### 0.5 PyTorch Fundamentals
**Learning Time**: 4-6 hours

**Concepts**:
- Tensors and operations
- Autograd basics
- GPU tensors (.cuda(), .to('cuda'))
- Custom operators overview
- Basic neural network

**Materials**:
- PyTorch quickstart
- GPU acceleration basics
- Integration with CUDA

**Outcomes**:
- [ ] Create and manipulate tensors
- [ ] Move tensors to/from GPU
- [ ] Understand autograd
- [ ] Train simple neural network

---

### L0 Capstone Project
**"Hello GPU World" Pipeline**

Build a complete pipeline:
1. Load data with PyTorch
2. Implement simple CUDA kernel (element-wise operation)
3. Create Python binding
4. Integrate with PyTorch
5. Benchmark vs CPU

**Time**: 4-6 hours
**Skills Validated**: Setup, basic CUDA, PyTorch integration

---

## L0 Assessment

- [ ] All topics completed
- [ ] Capstone project working
- [ ] Can compile and run CUDA programs
- [ ] Understand basic GPU concepts
- [ ] Ready for L1

**Estimated Total Time**: 1-2 weeks (15-25 hours)

---

# Level 1: Basic Kernels

## Overview
**Duration**: 2-3 weeks
**Prerequisite**: L0 complete
**Goal**: Write and optimize simple CUDA kernels

## Topics

### 1.1 Thread Hierarchy Deep Dive
**Learning Time**: 3-4 hours

**Concepts**:
- Grid dimensions and block dimensions
- Thread indexing (threadIdx, blockIdx, blockDim, gridDim)
- 1D, 2D, 3D configurations
- Choosing optimal block sizes
- Maximum blocks/threads limits

**Materials**:
- Interactive thread indexing visualizer
- Examples with different configurations
- Exercises: Calculate global indices

**Outcomes**:
- [ ] Calculate global thread IDs for any configuration
- [ ] Choose appropriate grid/block sizes
- [ ] Understand thread scheduling
- [ ] Handle boundary conditions

**Code Examples**:
- Vector operations with different block sizes
- 2D matrix indexing
- 3D volume processing

---

### 1.2 Memory Model Basics
**Learning Time**: 4-5 hours

**Concepts**:
- Global memory access
- Memory allocation (cudaMalloc, cudaMemcpy)
- Host-device transfers
- Memory bandwidth
- Coalesced access (introduction)

**Materials**:
- Memory hierarchy diagram
- Transfer timing examples
- Bandwidth measurements

**Outcomes**:
- [ ] Allocate GPU memory
- [ ] Transfer data between host and device
- [ ] Understand memory bandwidth importance
- [ ] Measure transfer times

**Code Examples**:
- Memory allocation patterns
- Pinned vs pageable memory
- Async transfers

---

### 1.3 Element-wise Operations
**Learning Time**: 3-4 hours

**Concepts**:
- Parallel element-wise patterns
- Activation functions (ReLU, GELU, Swish)
- Arithmetic operations
- Error handling
- Kernels vs PyTorch ops

**Materials**:
- Tutorial: "Implementing ReLU from Scratch"
- Benchmarks vs PyTorch
- Optimization basics

**Outcomes**:
- [ ] Implement activation functions
- [ ] Create PyTorch custom ops
- [ ] Benchmark kernel performance
- [ ] Understand overhead

**Code Examples**:
- ReLU, GELU, SiLU, Swish
- Vectorized operations
- Fused operations

---

### 1.4 Simple Reductions
**Learning Time**: 5-6 hours

**Concepts**:
- Reduction pattern
- Atomic operations
- Sequential addressing
- Warp divergence (introduction)
- Sum, max, min operations

**Materials**:
- Reduction algorithm explanation
- Step-by-step visualization
- Common pitfalls

**Outcomes**:
- [ ] Implement basic reduction
- [ ] Use atomic operations safely
- [ ] Understand warp divergence
- [ ] Measure reduction performance

**Code Examples**:
- Sum reduction
- Max/min finding
- Dot product
- L2 norm

---

### 1.5 Debugging and Profiling Basics
**Learning Time**: 3-4 hours

**Concepts**:
- cuda-memcheck
- printf debugging
- Error checking (cudaGetLastError)
- NSight Systems introduction
- Performance metrics

**Materials**:
- Debugging guide
- Common CUDA errors
- Profiling tutorial

**Outcomes**:
- [ ] Debug CUDA kernel errors
- [ ] Use cuda-memcheck
- [ ] Profile simple kernels
- [ ] Interpret profiling results

**Tools**:
- cuda-memcheck
- NSight Systems
- nvprof basics

---

### L1 Projects

#### Project 1.1: Custom Activation Library
**Time**: 6-8 hours

Implement complete activation function library:
- Forward: ReLU, LeakyReLU, GELU, Swish, SiLU, Mish
- Backward: All gradients
- PyTorch integration
- Benchmarks vs PyTorch
- Unit tests

**Skills**: Element-wise ops, PyTorch binding, testing

---

#### Project 1.2: Image Processing Pipeline
**Time**: 8-10 hours

Build image processing kernels:
- Brightness/contrast adjustment
- Saturation control
- Gaussian blur
- Sobel edge detection
- Benchmark vs CPU

**Skills**: 2D indexing, image data, memory transfers

---

#### Project 1.3: Vector Math Library
**Time**: 6-8 hours

Create optimized vector operations:
- Dot product, cross product
- Vector norms (L1, L2, Linf)
- Distance metrics (Euclidean, Manhattan, Cosine)
- Comparison with cuBLAS

**Skills**: Reductions, optimizations, benchmarking

---

## L1 Assessment

- [ ] All topics completed
- [ ] At least 2 projects completed
- [ ] Can write simple kernels independently
- [ ] Understand thread hierarchy
- [ ] Can debug and profile
- [ ] Ready for L2

**Estimated Total Time**: 2-3 weeks (30-45 hours)

---

# Level 2: Memory Optimization

## Overview
**Duration**: 3-4 weeks
**Prerequisite**: L1 complete
**Goal**: Master memory optimization techniques

## Topics

### 2.1 Memory Coalescing
**Learning Time**: 4-5 hours

**Concepts**:
- Memory transactions
- Coalesced vs uncoalesced access
- Alignment requirements
- Stride patterns
- Measuring coalescing efficiency

**Materials**:
- Visual demonstrations
- Profiling coalescing metrics
- Before/after examples

**Outcomes**:
- [ ] Identify uncoalesced access patterns
- [ ] Optimize memory access
- [ ] Measure memory efficiency
- [ ] Understand transaction overhead

**Code Examples**:
- Matrix transpose (naive vs coalesced)
- Array of structures vs structure of arrays
- Strided access patterns

---

### 2.2 Shared Memory
**Learning Time**: 6-8 hours

**Concepts**:
- Shared memory purpose and limits
- Declaration and usage
- __syncthreads() synchronization
- Shared memory as cache
- Tiling strategies

**Materials**:
- Shared memory tutorial
- Tiling visualizations
- Performance comparisons

**Outcomes**:
- [ ] Use shared memory effectively
- [ ] Implement tiling
- [ ] Synchronize threads correctly
- [ ] Measure shared memory benefit

**Code Examples**:
- Matrix multiplication with tiling
- 1D stencil with shared memory
- Histogram computation

---

### 2.3 Bank Conflicts
**Learning Time**: 4-5 hours

**Concepts**:
- Shared memory banks
- Bank conflict detection
- Padding strategies
- Conflict-free patterns
- Profiling bank conflicts

**Materials**:
- Bank conflict explanation
- Visual bank mapping
- Optimization techniques

**Outcomes**:
- [ ] Understand bank organization
- [ ] Detect bank conflicts in code
- [ ] Apply padding to avoid conflicts
- [ ] Verify conflict-free access

**Code Examples**:
- Matrix transpose with padding
- Conflict-free reductions
- Bank-friendly access patterns

---

### 2.4 Memory Access Patterns
**Learning Time**: 5-6 hours

**Concepts**:
- Sequential, strided, random access
- L1/L2 cache utilization
- Prefetching
- Memory bandwidth optimization
- Roofline model introduction

**Materials**:
- Access pattern comparisons
- Cache behavior analysis
- Bandwidth measurements

**Outcomes**:
- [ ] Design efficient access patterns
- [ ] Utilize cache effectively
- [ ] Maximize memory bandwidth
- [ ] Profile memory-bound kernels

**Code Examples**:
- Various access patterns
- Cache-aware algorithms
- Bandwidth optimization techniques

---

### 2.5 Advanced Reductions
**Learning Time**: 6-7 hours

**Concepts**:
- Tree-based reductions
- Warp shuffles
- Multiple blocks coordination
- Two-stage reductions
- CUB library introduction

**Materials**:
- Optimized reduction guide
- Warp primitives tutorial
- Performance analysis

**Outcomes**:
- [ ] Implement efficient reductions
- [ ] Use warp primitives
- [ ] Handle multi-block reductions
- [ ] Approach theoretical limits

**Code Examples**:
- Optimized sum reduction
- Warp-shuffle reductions
- Segmented reductions
- CUB comparisons

---

### 2.6 Occupancy Optimization
**Learning Time**: 4-5 hours

**Concepts**:
- Occupancy definition
- Register usage limits
- Shared memory limits
- Block size vs occupancy
- CUDA occupancy calculator

**Materials**:
- Occupancy guide
- Launch configuration tuning
- Profiling occupancy

**Outcomes**:
- [ ] Calculate occupancy
- [ ] Optimize launch configurations
- [ ] Balance resource usage
- [ ] Measure impact on performance

**Tools**:
- Occupancy calculator
- NSight Compute
- Manual calculation

---

### L2 Projects

#### Project 2.1: Optimized Convolution Engine
**Time**: 12-15 hours

Implement multiple convolution strategies:
- Naive direct convolution
- Im2col + GEMM
- Tiled convolution with shared memory
- Winograd (bonus)
- Benchmark all approaches

**Skills**: Tiling, shared memory, algorithm selection

---

#### Project 2.2: Softmax Kernels
**Time**: 8-10 hours

Implement softmax variants:
- Naive softmax
- Online softmax (numerical stable)
- Fused softmax (forward + backward)
- Support for different dimensions
- Comparison with PyTorch

**Skills**: Reductions, numerical stability, fusion

---

#### Project 2.3: Layer Normalization
**Time**: 10-12 hours

Complete LayerNorm implementation:
- Forward pass with Welford's algorithm
- Backward pass
- Fused implementation
- Affine transformation option
- Benchmarks vs PyTorch

**Skills**: Two-pass algorithms, fusion, optimization

---

## L2 Assessment

- [ ] All topics completed
- [ ] At least 2 projects completed
- [ ] Understand memory hierarchy thoroughly
- [ ] Can optimize memory-bound kernels
- [ ] Achieve >80% of theoretical bandwidth
- [ ] Ready for L3

**Estimated Total Time**: 3-4 weeks (50-70 hours)

---

# Level 3: Advanced Patterns

## Overview
**Duration**: 4-6 weeks
**Prerequisite**: L2 complete
**Goal**: Implement complex algorithms and achieve high performance

## Topics

### 3.1 Warp-Level Primitives
**Learning Time**: 6-7 hours

**Concepts**:
- Warp concept and scheduling
- Warp intrinsics (__shfl, __ballot, etc.)
- Cooperative groups
- Warp-level reductions
- Warp specialization

**Materials**:
- Warp primitives guide
- Shuffle pattern examples
- Cooperative groups tutorial

**Outcomes**:
- [ ] Use warp shuffle effectively
- [ ] Implement warp-level algorithms
- [ ] Understand warp divergence
- [ ] Optimize warp utilization

**Code Examples**:
- Warp-shuffle reduction
- Prefix sum with shuffles
- Warp-level matrix multiply
- Ballot and vote operations

---

### 3.2 GEMM Optimization
**Learning Time**: 12-15 hours

**Concepts**:
- GEMM algorithm and importance
- Tiling strategies
- Register blocking
- Vectorized loads/stores
- Achieving high FLOPS

**Materials**:
- GEMM tutorial series
- Optimization techniques
- Roofline analysis

**Outcomes**:
- [ ] Implement tiled GEMM
- [ ] Apply register blocking
- [ ] Reach 70%+ of cuBLAS
- [ ] Understand GEMM performance model

**Code Examples**:
- Naive GEMM
- Tiled GEMM (shared memory)
- Register-blocked GEMM
- Vectorized GEMM

**Progression**:
1. Naive (5-10% of cuBLAS)
2. Shared memory tiling (30-40%)
3. Register blocking (50-60%)
4. Vectorization + tuning (70-80%)

---

### 3.3 Attention Mechanisms
**Learning Time**: 10-12 hours

**Concepts**:
- Scaled dot-product attention
- Multi-head attention
- Causal masking
- Memory-efficient attention
- Softmax bottleneck

**Materials**:
- Attention algorithm explanation
- Implementation guide
- Flash Attention introduction

**Outcomes**:
- [ ] Implement basic attention
- [ ] Handle different head dimensions
- [ ] Support causal masking
- [ ] Understand memory challenges

**Code Examples**:
- Naive attention (QK^T, softmax, x V)
- Multi-head attention
- Causal attention
- Memory analysis

---

### 3.4 Positional Encodings
**Learning Time**: 4-5 hours

**Concepts**:
- Absolute vs relative positions
- Sinusoidal encodings
- RoPE (Rotary Position Embeddings)
- ALiBi
- Implementation techniques

**Materials**:
- Position encoding tutorial
- RoPE explanation
- Efficient implementations

**Outcomes**:
- [ ] Implement RoPE
- [ ] Understand position encoding goals
- [ ] Optimize encoding kernels
- [ ] Integrate with attention

**Code Examples**:
- Sinusoidal embeddings
- RoPE implementation
- Fused RoPE + attention

---

### 3.5 Advanced Normalization
**Learning Time**: 5-6 hours

**Concepts**:
- BatchNorm vs LayerNorm vs RMSNorm
- Welford's algorithm
- Numerically stable implementations
- Affine transformations
- Fusion opportunities

**Materials**:
- Normalization comparison
- Numerical stability guide
- Fusion strategies

**Outcomes**:
- [ ] Implement all normalization types
- [ ] Ensure numerical stability
- [ ] Fuse with adjacent operations
- [ ] Handle different input shapes

**Code Examples**:
- BatchNorm (training + inference)
- RMSNorm (LLaMA-style)
- Fused normalization + activation

---

### 3.6 Kernel Fusion
**Learning Time**: 6-7 hours

**Concepts**:
- Why fuse kernels
- Fusion opportunities
- Memory vs compute trade-offs
- Common fusion patterns
- Integration with frameworks

**Materials**:
- Fusion guide
- Pattern identification
- Performance analysis

**Outcomes**:
- [ ] Identify fusion opportunities
- [ ] Implement fused kernels
- [ ] Measure fusion benefits
- [ ] Understand trade-offs

**Code Examples**:
- Bias + ReLU
- GEMM + Bias + GELU
- Normalization + Residual
- Attention + Projection

---

### L3 Projects

#### Project 3.1: GEMM Library
**Time**: 20-25 hours

Complete matrix multiplication library:
- Multiple optimization levels
- Support FP32, FP16, mixed precision
- Auto-tuning for different shapes
- Reach 80%+ of cuBLAS
- Python bindings

**Skills**: GEMM optimization, tuning, comprehensive testing

---

#### Project 3.2: Attention Module
**Time**: 18-22 hours

Complete attention implementation:
- Multi-head attention
- Support for different head dimensions
- Causal and non-causal
- Forward and backward
- RoPE integration
- Benchmark vs PyTorch SDPA

**Skills**: Attention algorithms, memory management, autograd

---

#### Project 3.3: Transformer Layer
**Time**: 25-30 hours

Full transformer decoder layer:
- Multi-head attention
- Feed-forward network (GEMM + activation)
- Layer normalization
- Residual connections
- Multiple fusion strategies
- Complete forward/backward

**Skills**: Integration, fusion, end-to-end optimization

---

## L3 Assessment

- [ ] All topics completed
- [ ] At least 2 projects completed
- [ ] GEMM achieves 70%+ of cuBLAS
- [ ] Understand complex algorithms
- [ ] Can design fused kernels
- [ ] Ready for L4

**Estimated Total Time**: 4-6 weeks (70-100 hours)

---

# Level 4: Tensor Cores & Advanced

## Overview
**Duration**: 6-8 weeks
**Prerequisite**: L3 complete
**Goal**: Master Tensor Cores and advanced optimization techniques

## Topics

### 4.1 Tensor Core Programming (WMMA)
**Learning Time**: 8-10 hours

**Concepts**:
- Tensor Core architecture
- WMMA API (Warp Matrix Multiply-Accumulate)
- Fragment types
- Matrix shapes (m16n16k16, etc.)
- FP16, BF16, TF32, INT8

**Materials**:
- Tensor Core architecture guide
- WMMA tutorial
- Fragment management

**Outcomes**:
- [ ] Understand Tensor Core capabilities
- [ ] Use WMMA API
- [ ] Implement WMMA GEMM
- [ ] Support multiple data types

**Code Examples**:
- Basic WMMA usage
- WMMA HGEMM
- Mixed precision patterns
- Performance analysis

---

### 4.2 PTX and MMA Instructions
**Learning Time**: 10-12 hours

**Concepts**:
- PTX (Parallel Thread Execution)
- MMA (Matrix Multiply-Accumulate) instructions
- Inline assembly in CUDA
- Register management
- Instruction-level control

**Materials**:
- PTX ISA reference (subset)
- MMA instruction guide
- Assembly examples

**Outcomes**:
- [ ] Read PTX code
- [ ] Use inline PTX assembly
- [ ] Implement MMA-based GEMM
- [ ] Understand low-level control

**Code Examples**:
- MMA instruction usage
- PTX HGEMM
- Register optimization
- Instruction pipelining

---

### 4.3 CUTLASS Introduction
**Learning Time**: 8-10 hours

**Concepts**:
- CUTLASS library overview
- Tile iterators
- CuTe (CUTLASS template library)
- Layout abstractions
- GEMM using CUTLASS

**Materials**:
- CUTLASS documentation
- CuTe tutorial
- Example analyses

**Outcomes**:
- [ ] Understand CUTLASS architecture
- [ ] Use CuTe layouts
- [ ] Implement GEMM with CUTLASS
- [ ] Leverage existing optimizations

**Code Examples**:
- CuTe layout basics
- CUTLASS GEMM template
- Custom CUTLASS kernels

---

### 4.4 Flash Attention Deep Dive
**Learning Time**: 12-15 hours

**Concepts**:
- IO-awareness
- Tiling for SRAM
- Online softmax computation
- Flash Attention 1 vs 2
- Forward and backward passes

**Materials**:
- Flash Attention papers
- Algorithm walkthrough
- Implementation guide

**Outcomes**:
- [ ] Understand Flash Attention algorithm
- [ ] Implement forward pass
- [ ] Implement backward pass
- [ ] Achieve 90%+ of reference

**Code Examples**:
- Flash Attention V1 (split-KV)
- Flash Attention V2 (split-Q)
- Shared KV optimization
- QKV tiling

**Projects**:
- Complete Flash Attention implementation
- Support multiple head dimensions
- Benchmark vs PyTorch SDPA

---

### 4.5 Advanced Fusion and Pipelining
**Learning Time**: 8-10 hours

**Concepts**:
- Software pipelining
- Multi-stage pipelines
- Async copy (cp.async)
- Double/triple buffering
- Latency hiding

**Materials**:
- Pipelining guide
- Async copy tutorial
- Performance modeling

**Outcomes**:
- [ ] Implement pipelined kernels
- [ ] Use async copy instructions
- [ ] Hide memory latency
- [ ] Measure pipeline efficiency

**Code Examples**:
- Double-buffered GEMM
- Triple-buffered pipeline
- Async copy patterns
- Prefetching strategies

---

### 4.6 Quantization
**Learning Time**: 6-8 hours

**Concepts**:
- INT8, INT4 quantization
- Per-tensor vs per-channel
- Quantization-aware training
- Dequantization strategies
- DP4A instructions (INT8)

**Materials**:
- Quantization overview
- INT8 GEMM tutorial
- Accuracy vs performance

**Outcomes**:
- [ ] Implement INT8 GEMM
- [ ] Understand quantization schemes
- [ ] Measure accuracy impact
- [ ] Optimize INT8 kernels

**Code Examples**:
- INT8 matrix multiply
- Dynamic quantization
- Mixed precision inference

---

### L4 Projects

#### Project 4.1: Production GEMM Library
**Time**: 30-40 hours

Full-featured GEMM library:
- WMMA and MMA implementations
- Support FP32, TF32, FP16, BF16, INT8
- Auto-tuning infrastructure
- Reach 95%+ of cuBLAS
- Comprehensive benchmarks
- Python bindings

**Skills**: Tensor Cores, optimization, auto-tuning

---

#### Project 4.2: Flash Attention Library
**Time**: 35-45 hours

Complete Flash Attention implementation:
- Forward and backward passes
- Multiple head dimensions (64, 128, 256)
- Causal and non-causal
- Variable sequence lengths
- Dropout support (bonus)
- Match FlashAttention-2 performance

**Skills**: Advanced algorithms, memory optimization, Tensor Cores

---

#### Project 4.3: Quantized Inference Engine
**Time**: 25-30 hours

INT8 inference system:
- Quantized GEMM
- Quantized attention
- Dynamic quantization
- Calibration utilities
- Example models (BERT, GPT)
- Accuracy validation

**Skills**: Quantization, INT8 ops, inference optimization

---

## L4 Assessment

- [ ] All topics completed
- [ ] At least 2 projects completed
- [ ] GEMM achieves 90%+ of cuBLAS
- [ ] Understand Tensor Cores thoroughly
- [ ] Can implement research papers
- [ ] Ready for L5

**Estimated Total Time**: 6-8 weeks (100-140 hours)

---

# Level 5: Production Systems

## Overview
**Duration**: 8-12 weeks
**Prerequisite**: L4 complete
**Goal**: Build production-ready, multi-GPU systems

## Topics

### 5.1 Multi-GPU Programming
**Learning Time**: 10-12 hours

**Concepts**:
- Multi-GPU architectures
- Peer-to-peer access
- Unified memory
- Streams and events
- GPU selection strategies

**Materials**:
- Multi-GPU guide
- Topology understanding
- Scaling patterns

**Outcomes**:
- [ ] Manage multiple GPUs
- [ ] Optimize P2P transfers
- [ ] Implement multi-GPU algorithms
- [ ] Handle GPU topology

**Code Examples**:
- Multi-GPU GEMM
- Data parallelism
- Model parallelism basics
- P2P bandwidth tests

---

### 5.2 NCCL and Collective Operations
**Learning Time**: 8-10 hours

**Concepts**:
- NCCL library
- AllReduce, AllGather, ReduceScatter
- Ring algorithms
- Tree algorithms
- Communicator management

**Materials**:
- NCCL documentation
- Collective patterns
- Performance tuning

**Outcomes**:
- [ ] Use NCCL effectively
- [ ] Implement distributed algorithms
- [ ] Optimize communication
- [ ] Handle failures

**Code Examples**:
- NCCL AllReduce
- Distributed training primitives
- Gradient synchronization
- Custom collectives

---

### 5.3 Distributed Training Patterns
**Learning Time**: 12-15 hours

**Concepts**:
- Data parallelism (DP, DDP)
- Model parallelism (MP)
- Pipeline parallelism (PP)
- Tensor parallelism (TP)
- ZeRO optimizations

**Materials**:
- Parallelism strategies guide
- Megatron-LM patterns
- DeepSpeed ZeRO

**Outcomes**:
- [ ] Implement data parallelism
- [ ] Understand model parallelism
- [ ] Design hybrid strategies
- [ ] Optimize communication/compute overlap

**Code Examples**:
- DDP implementation
- Tensor parallelism for transformer
- Pipeline parallelism
- Gradient accumulation

---

### 5.4 Operator Fusion Frameworks
**Learning Time**: 10-12 hours

**Concepts**:
- Automatic fusion
- Pattern matching
- Code generation
- JIT compilation
- Integration with PyTorch/JAX

**Materials**:
- Fusion framework design
- TorchScript, XLA
- Custom fusion examples

**Outcomes**:
- [ ] Design fusion systems
- [ ] Implement pattern matching
- [ ] Generate fused kernels
- [ ] Measure fusion benefits

**Code Examples**:
- Simple fusion framework
- Pattern matching engine
- Code generation
- Integration examples

---

### 5.5 Auto-Tuning Systems
**Learning Time**: 10-12 hours

**Concepts**:
- Parameter search spaces
- Auto-tuning strategies
- Performance modeling
- Result caching
- Online vs offline tuning

**Materials**:
- Auto-tuning guide
- Search algorithms
- Tuning frameworks (OpenTuner, etc.)

**Outcomes**:
- [ ] Design tuning infrastructure
- [ ] Implement parameter search
- [ ] Build performance models
- [ ] Deploy tuned kernels

**Code Examples**:
- Grid search auto-tuner
- Genetic algorithm tuner
- Cache management
- Runtime selection

---

### 5.6 Production Deployment
**Learning Time**: 8-10 hours

**Concepts**:
- PyTorch custom extensions
- Packaging and distribution
- Version compatibility
- Error handling and logging
- Performance monitoring
- CI/CD for GPU code

**Materials**:
- Extension packaging guide
- Deployment best practices
- Monitoring strategies

**Outcomes**:
- [ ] Package custom operators
- [ ] Handle version compatibility
- [ ] Implement robust error handling
- [ ] Monitor production performance

**Code Examples**:
- setup.py for CUDA extensions
- Version checking
- Graceful degradation
- Performance monitoring

---

### L5 Projects

#### Project 5.1: Distributed Training Framework
**Time**: 40-50 hours

Complete training framework:
- Data parallelism with NCCL
- Model parallelism support
- Gradient accumulation
- Mixed precision training
- Checkpointing and resume
- Multi-node support
- Example training scripts

**Skills**: Distributed systems, NCCL, production readiness

---

#### Project 5.2: Inference Engine
**Time**: 35-45 hours

Production inference system:
- Continuous batching
- KV cache management
- Dynamic quantization
- Multi-GPU inference
- Request scheduling
- Monitoring and metrics
- Example serving API

**Skills**: Inference optimization, system design, deployment

---

#### Project 5.3: End-to-End ML Pipeline
**Time**: 45-60 hours

Complete ML system:
- Custom operators throughout
- Training with custom kernels
- Distributed training
- Model export and optimization
- Inference deployment
- Monitoring and profiling
- Documentation

**Skills**: Integration, end-to-end optimization, production

---

## L5 Assessment

- [ ] All topics completed
- [ ] At least 2 projects completed
- [ ] Can build production systems
- [ ] Understand distributed training
- [ ] Deployment experience
- [ ] Expert-level GPU programmer

**Estimated Total Time**: 8-12 weeks (120-180 hours)

---

# Learning Tracks

## Fast Track (6-8 months)

**Target**: Experienced programmers, 15-20 hours/week

**Path**:
- L0: 1 week (skip if familiar)
- L1: 2 weeks
- L2: 3 weeks
- L3: 5 weeks
- L4: 7 weeks
- L5: 10 weeks

**Focus**:
- Core concepts only
- 1 project per level
- Optimization-heavy

---

## Standard Track (12-18 months)

**Target**: Regular pace, 8-12 hours/week

**Path**:
- L0: 2 weeks
- L1: 3 weeks
- L2: 4 weeks
- L3: 6 weeks
- L4: 8 weeks
- L5: 12 weeks

**Focus**:
- All topics thoroughly
- 2-3 projects per level
- Balanced theory and practice

---

## Deep Dive Track (18-24 months)

**Target**: Research-oriented, comprehensive

**Path**:
- L0: 2 weeks
- L1: 4 weeks
- L2: 6 weeks
- L3: 8 weeks
- L4: 12 weeks
- L5: 16 weeks
- Additional paper implementations
- Original research projects

**Focus**:
- All topics + extensions
- All projects + bonus challenges
- Paper implementations
- Novel contributions

---

# Prerequisites and Skill Dependencies

```
L0: Foundation
├── None (programming basics assumed)
│
L1: Basic Kernels
├── L0: Complete
│
L2: Memory Optimization
├── L1: Complete
├── Specific: Thread hierarchy, basic kernels
│
L3: Advanced Patterns
├── L2: Complete
├── Specific: Shared memory, coalescing, reductions
│
L4: Tensor Cores
├── L3: Complete
├── Specific: GEMM optimization, attention basics
│
L5: Production Systems
├── L4: Complete
├── Specific: Tensor Cores, complex algorithms
└── Optional: Distributed systems knowledge
```

---

# Assessment and Certification

## Self-Assessment Criteria

### After Each Level:
1. **Knowledge Check**: Answer conceptual questions
2. **Coding Challenge**: Implement kernel from spec
3. **Optimization Task**: Improve given kernel
4. **Project Review**: Complete assigned projects

### Mastery Indicators:
- **L1**: Can write simple kernels independently
- **L2**: Achieves >80% memory bandwidth
- **L3**: GEMM reaches >70% of cuBLAS
- **L4**: Implements research papers
- **L5**: Deploys production systems

---

# Resources

## For Each Level:

### Readings
- Curated papers
- NVIDIA documentation
- Blog posts and tutorials

### Code Examples
- Starter templates
- Progressive implementations
- Optimization stages

### Exercises
- Guided modifications
- Performance challenges
- Debugging scenarios

### Projects
- Specifications
- Starter code
- Reference solutions
- Grading rubrics

---

# Continuous Learning

## Stay Updated:
- New CUDA features (CUDA 12.x+)
- Latest research papers
- Emerging frameworks (Triton, Mojo, etc.)
- Hardware advances (Hopper, Blackwell, etc.)

## Community Engagement:
- Contribute to LeetCUDA
- Share implementations
- Discuss optimizations
- Help fellow learners

---

**Version**: 1.0
**Last Updated**: 2025-11-19
**Next Review**: Quarterly updates
