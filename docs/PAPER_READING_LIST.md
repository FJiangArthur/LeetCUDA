# LeetCUDA Paper Reading List

## Overview

This document lists essential research papers for GPU programming and deep learning optimization. Each paper includes a summary, implementation status, and recommended reading level.

---

## Legend

- **Level**: L1-L5 (when to read based on your progress)
- **Status**: 📝 Summary Available | 💻 Implementation Available | ⏳ Planned
- **Difficulty**: ⭐ Easy | ⭐⭐ Medium | ⭐⭐⭐ Hard | ⭐⭐⭐⭐ Expert

---

## Foundational

### GPU Computing Basics

#### 1. CUDA Programming Model
**Authors**: NVIDIA
**Published**: 2007 (Updated continuously)
**Level**: L0
**Difficulty**: ⭐
**Status**: 📝💻

**Why Read**: Foundation of CUDA programming

**Summary**: Introduces the CUDA programming model, thread hierarchy, memory model, and basic concepts.

**Key Takeaways**:
- SIMT execution model
- Thread, block, grid hierarchy
- Memory spaces (global, shared, local)
- Kernel execution model

**Implementation in LeetCUDA**: `learning-paths/L0-foundation/`

---

## Matrix Operations

### 2. Optimizing Matrix Multiplication (GEMM)
**Authors**: Volkov & Demmel
**Published**: 2008
**Conference**: SC '08
**Level**: L2-L3
**Difficulty**: ⭐⭐⭐
**Status**: 📝💻

**Why Read**: Core algorithm for deep learning, optimization techniques applicable broadly

**Summary**: Comprehensive study of matrix multiplication optimization on GPUs, covering tiling, register blocking, and achieving high performance.

**Key Innovations**:
- Hierarchical tiling (registers, shared memory, global memory)
- Register blocking for reduced shared memory access
- Warp-level optimization techniques

**Implementation**: `kernels/sgemm/`, `kernels/hgemm/`

**Related Papers**:
- "A Fast Method for Matrix Balancing" (2016)
- "Communication-Optimal Parallel Algorithm for Strassen's Matrix Multiplication" (2012)

---

### 3. CUTLASS: Fast Linear Algebra in CUDA C++
**Authors**: NVIDIA
**Published**: 2018 (Ongoing)
**Level**: L4
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝💻

**Why Read**: Production-ready template library for high-performance GEMM

**Summary**: Template-based library for GEMM and related operations, emphasizing performance, composability, and extensibility.

**Key Concepts**:
- Tile iterators
- Layout abstractions
- CuTe (CUTLASS Template Library)
- Warp-specialized operations

**Implementation**: `kernels/cutlass/`

**Prerequisites**: L3 GEMM understanding, C++ templates

---

## Attention Mechanisms

### 4. Attention Is All You Need
**Authors**: Vaswani et al. (Google)
**Published**: 2017
**Conference**: NeurIPS
**Level**: L2-L3
**Difficulty**: ⭐⭐
**Status**: 📝💻

**Why Read**: Foundation of modern transformers, understand the algorithm before optimizing

**Summary**: Introduces the Transformer architecture with self-attention as the core mechanism, replacing recurrence and convolution.

**Key Concepts**:
- Scaled dot-product attention
- Multi-head attention
- Positional encoding
- Feed-forward networks

**Attention Formula**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**Implementation**: `kernels/transformer/attention/`, `models/transformers/`

**Related Papers**:
- "BERT" (2018)
- "GPT" series (2018-2023)

---

### 5. Flash Attention: Fast and Memory-Efficient Exact Attention
**Authors**: Dao et al. (Stanford)
**Published**: 2022
**Conference**: NeurIPS
**Level**: L4
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝💻

**Why Read**: Breakthrough in attention efficiency, demonstrates IO-aware algorithm design

**Summary**: Introduces IO-aware exact attention algorithm that reduces HBM accesses by recomputing attention on-the-fly, achieving 2-4x speedup.

**Key Innovations**:
- **Tiling**: Split Q, K, V into blocks fitting in SRAM
- **Online Softmax**: Compute softmax incrementally without storing QK^T
- **Recomputation**: Trade compute for memory in backward pass

**Algorithm Outline**:
```
For each block of Q:
  Load Q_i to SRAM
  Initialize O_i = 0, l_i = 0, m_i = -inf

  For each block of K, V:
    Load K_j, V_j to SRAM
    Compute S_ij = Q_i @ K_j^T on-chip
    Update statistics (m, l) for online softmax
    Update O_i with contribution from V_j

  Write O_i to HBM
```

**Performance**:
- 2-4x faster than PyTorch SDPA on A100
- Sub-quadratic memory (O(n) vs O(n²))
- Enables longer sequences

**Implementation**: `kernels/flash-attn/flash_attn_v1.cu`

**Prerequisites**: L3 attention, L2 tiling, online algorithms

---

### 6. Flash Attention 2: Faster Attention with Better Parallelism
**Authors**: Dao (Stanford)
**Published**: 2023
**Level**: L4
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝💻

**Why Read**: Improves upon Flash Attention with better parallelism and hardware utilization

**Summary**: Refines Flash Attention algorithm with improved parallelism strategy (split-Q instead of split-KV), reducing synchronization and improving performance.

**Key Improvements over V1**:
- **Split-Q strategy**: Parallelize over Q blocks instead of KV
- **Reduced synchronization**: Fewer atomic operations
- **Better warp utilization**: More work per warp
- **Improved backward pass**: More efficient gradient computation

**Performance**:
- 1.5-2x faster than Flash Attention 1
- 90-95% of theoretical max FLOPs
- Scales to longer sequences

**Implementation**: `kernels/flash-attn/flash_attn_v2.cu`

**Related Papers**:
- "Flash Attention 3" (2024) - Hardware-specific optimizations

---

### 7. Self-attention Does Not Need O(n²) Memory
**Authors**: Rabe & Staats (Google)
**Published**: 2021
**Level**: L4
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝

**Why Read**: Theoretical foundation for memory-efficient attention, precursor to Flash Attention

**Summary**: Proves that self-attention can be computed with O(n) memory in the backward pass through recomputation.

**Key Insights**:
- Backward pass doesn't need to store QK^T
- Recomputation is more efficient than storing
- Theoretical analysis of memory-compute trade-offs

**Impact**: Inspired Flash Attention and related work

---

### 8. Paged Attention (vLLM)
**Authors**: Kwon et al. (UC Berkeley)
**Published**: 2023
**Conference**: SOSP
**Level**: L5
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝💻

**Why Read**: Efficient KV cache management for LLM inference

**Summary**: Applies virtual memory concepts to attention KV cache, enabling dynamic memory allocation and sharing.

**Key Innovations**:
- **Paged KV cache**: Split KV into fixed-size blocks
- **Virtual memory**: Logical to physical block mapping
- **Copy-on-write**: Share KV between sequences (beam search)
- **Continuous batching**: Dynamic batch composition

**Performance**:
- 2-4x higher throughput than static batching
- Near-zero memory waste
- Enables larger batch sizes

**Implementation**: `models/inference/paged_attention/`

**Prerequisites**: L4 attention, L5 systems concepts

---

## Memory Optimization

### 9. Mixed Precision Training
**Authors**: Micikevicius et al. (NVIDIA)
**Published**: 2018
**Conference**: ICLR
**Level**: L2-L3
**Difficulty**: ⭐⭐
**Status**: 📝💻

**Why Read**: Essential technique for modern deep learning

**Summary**: Demonstrates that training with FP16 weights and activations (with FP32 master weights and loss scaling) achieves same accuracy as FP32 while being faster and using less memory.

**Key Techniques**:
- FP16 storage and computation
- FP32 master weights
- Loss scaling to prevent underflow
- Accumulation in FP32

**Implementation**: `examples/mixed_precision/`

---

### 10. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
**Authors**: Rajbhandari et al. (Microsoft)
**Published**: 2020
**Conference**: SC
**Level**: L5
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝

**Why Read**: State-of-the-art for large-scale distributed training

**Summary**: Introduces ZeRO optimizer that partitions optimizer states, gradients, and parameters across devices to dramatically reduce memory footprint.

**ZeRO Stages**:
- **Stage 1**: Partition optimizer states → 4x memory reduction
- **Stage 2**: + Partition gradients → 8x memory reduction
- **Stage 3**: + Partition parameters → Linear scaling with devices

**Impact**: Enables training models 100x larger on same hardware

**Related Papers**:
- "ZeRO-Offload" (2021)
- "ZeRO-Infinity" (2021)

---

## Distributed Training

### 11. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
**Authors**: Shoeybi et al. (NVIDIA)
**Published**: 2019
**Level**: L5
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝💻

**Why Read**: Foundational work on tensor and pipeline parallelism

**Summary**: Introduces efficient model parallelism strategies for training very large language models.

**Key Techniques**:
- **Tensor Parallelism**: Split tensors across GPUs within a layer
- **Pipeline Parallelism**: Split layers across GPUs
- **Optimized GEMM partitioning**: Minimize communication
- **Column-parallel vs row-parallel layers**

**Implementation**: `examples/distributed/megatron_patterns/`

**Prerequisites**: L4 complete, distributed systems basics

---

### 12. NCCL: Optimized Primitives for Collective Communication
**Authors**: NVIDIA
**Level**: L5
**Difficulty**: ⭐⭐⭐
**Status**: 📝💻

**Why Read**: Essential library for multi-GPU training

**Summary**: High-performance collective communication library optimized for GPUs.

**Collective Operations**:
- **AllReduce**: Sum gradients across all GPUs
- **AllGather**: Gather data from all GPUs
- **ReduceScatter**: Reduce and distribute
- **Broadcast**: Send from one to all

**Algorithms**:
- Ring algorithms (bandwidth optimal)
- Tree algorithms (latency optimal)
- Topology-aware routing

**Implementation**: `examples/distributed/nccl_examples/`

---

## Optimization Techniques

### 13. Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions
**Authors**: Vasilache et al. (Facebook)
**Published**: 2018
**Conference**: arXiv
**Level**: L4-L5
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝

**Why Read**: Automatic kernel generation from high-level specifications

**Summary**: Domain-specific language and compiler for automatically generating optimized CUDA kernels from tensor operations.

**Key Concepts**:
- DSL for tensor operations
- Polyhedral compilation
- Auto-tuning

**Impact**: Inspired Triton and other kernel generators

---

### 14. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations
**Authors**: Tillet et al. (OpenAI)
**Published**: 2019 (Updated 2021)
**Level**: L4
**Difficulty**: ⭐⭐⭐
**Status**: 📝💻

**Why Read**: Modern approach to kernel development, Python-based

**Summary**: Python-embedded DSL for writing GPU kernels at a higher level than CUDA, with automatic optimization.

**Key Features**:
- Python-like syntax
- Automatic memory coalescing
- Automatic shared memory management
- JIT compilation
- Performance comparable to hand-written CUDA

**Implementation**: `kernels/openai-triton/`

**Prerequisites**: L3 optimization understanding

---

## Quantization & Compression

### 15. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
**Authors**: Dettmers et al.
**Published**: 2022
**Conference**: NeurIPS
**Level**: L4
**Difficulty**: ⭐⭐⭐
**Status**: 📝💻

**Why Read**: Practical quantization for large models

**Summary**: Demonstrates that transformers can use INT8 matrix multiplication for most operations without accuracy loss through vector-wise quantization and outlier detection.

**Key Techniques**:
- Vector-wise quantization
- Outlier detection and mixed precision
- Two-stage decomposition
- Zero degradation for models up to 175B parameters

**Implementation**: `kernels/quantization/int8_gemm.cu`

---

### 16. SmoothQuant: Accurate and Efficient Post-Training Quantization
**Authors**: Xiao et al. (MIT)
**Published**: 2023
**Conference**: ICML
**Level**: L4
**Difficulty**: ⭐⭐⭐
**Status**: 📝

**Why Read**: Improved quantization with channel-wise scaling

**Summary**: Migrates difficulty from activations to weights through online channel-wise scaling, enabling more efficient INT8 quantization.

**Key Innovation**: Activation and weight smoothing to balance quantization difficulty

---

## Advanced Topics

### 17. Ring Attention: Blockwise Transformers for Near-Infinite Context
**Authors**: Liu & Abbeel (Berkeley)
**Published**: 2023
**Level**: L5
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝

**Why Read**: Extending context length beyond hardware limits

**Summary**: Distributes attention computation across devices using ring communication pattern, enabling extremely long contexts.

**Key Insights**:
- Overlap communication and computation
- Process sequence chunks in ring fashion
- Enables million-token contexts

---

### 18. FlashDecoding++: Faster Large Language Model Inference with Asynchronous Partial Attention
**Authors**: Hong et al. (SJTU & ETH Zurich)
**Published**: 2023
**Level**: L5
**Difficulty**: ⭐⭐⭐⭐
**Status**: ⏳

**Why Read**: State-of-the-art LLM inference optimization

**Summary**: Asynchronous partial attention computation for faster autoregressive decoding.

---

### 19. Continuous Batching (Orca)
**Authors**: Yu et al. (Microsoft)
**Published**: 2022
**Conference**: OSDI
**Level**: L5
**Difficulty**: ⭐⭐⭐
**Status**: 📝

**Why Read**: Essential for inference serving systems

**Summary**: Iteration-level batching instead of request-level, dramatically improving throughput and latency.

**Key Insight**: Add/remove requests from batch at each iteration rather than waiting for all to complete

---

### 20. Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2Seq Generation
**Authors**: Leviathan et al. (Google)
**Published**: 2023
**Level**: L5
**Difficulty**: ⭐⭐⭐⭐
**Status**: 📝

**Why Read**: Cutting-edge inference acceleration

**Summary**: Use small model to generate candidate tokens, verify with large model in parallel, achieving 2-3x speedup.

---

## How to Use This List

### By Learning Level

**L0 (Foundation)**:
- Papers 1

**L1 (Basic Kernels)**:
- Review Paper 1 (CUDA model)

**L2 (Memory Optimization)**:
- Papers 2, 9

**L3 (Advanced Patterns)**:
- Papers 2, 4

**L4 (Tensor Cores)**:
- Papers 3, 5, 6, 7, 14, 15

**L5 (Production)**:
- Papers 8, 10, 11, 12, 17-20

### By Interest Area

**GEMM Optimization**:
Papers 2, 3

**Attention Mechanisms**:
Papers 4, 5, 6, 7, 8, 17

**Distributed Training**:
Papers 10, 11, 12

**Quantization**:
Papers 15, 16

**Inference Serving**:
Papers 8, 18, 19, 20

**Kernel Generation**:
Papers 13, 14

---

## Reading Strategy

### Progressive Reading

1. **Skim First**: Get main ideas (15 min)
2. **Deep Read**: Understand algorithm (1-2 hours)
3. **Implementation**: Code it up (varies)
4. **Experimentation**: Try variations (1-2 hours)
5. **Related Work**: Read references (ongoing)

### Focus Areas

For each paper:
- **Problem**: What challenge does it address?
- **Innovation**: What's the key idea?
- **Algorithm**: How does it work?
- **Implementation**: How to code it?
- **Performance**: What speedups achieved?
- **Limitations**: When does it not work?

---

## Contributing

To add a paper:
1. Read and understand the paper
2. Write summary using template (see `docs/templates/paper_summary.md`)
3. Implement if applicable
4. Add to this list with appropriate metadata
5. Create PR

---

## Additional Resources

### Paper Repositories
- [Papers with Code - CUDA](https://paperswithcode.com/search?q_meta=&q_type=&q=cuda)
- [Awesome-CUDA](https://github.com/topics/cuda-programming)
- [GPU Gems](https://developer.nvidia.com/gpugems/gpugems3/contributors)

### Conferences to Follow
- **NeurIPS**: ML algorithms and systems
- **ICML**: Machine learning
- **ICLR**: Deep learning
- **SC**: Supercomputing and HPC
- **OSDI/SOSP**: Operating systems and design
- **MLSys**: ML systems
- **PPoPP**: Parallel programming

---

**Version**: 1.0
**Last Updated**: 2025-11-19
**Papers Count**: 20 core + many related
**Next Update**: Quarterly
