# Flash Attention: Fast and Memory-Efficient Exact Attention

## Metadata
- **Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford University)
- **Published**: NeurIPS 2022
- **Paper**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **Code**: [github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
- **Level**: L4 (Expert)
- **Difficulty**: ⭐⭐⭐⭐
- **Implementation Status**: 💻 Available in `kernels/flash-attn/`

---

## TL;DR

Flash Attention is an **IO-aware exact attention algorithm** that reduces memory reads/writes by recomputing the attention matrix on-the-fly from SRAM, achieving **2-4x speedup** and **sub-quadratic memory** compared to standard attention while being mathematically equivalent.

**Key Innovation**: Trade redundant HBM accesses for cheaper recomputation, using tiling and online softmax.

---

## The Problem

### Standard Attention is Memory-Bound

Standard attention implementation:
```python
# Q, K, V: [batch, seqlen, headdim]
S = Q @ K.T  # [batch, seqlen, seqlen] - HUGE!
P = softmax(S, dim=-1)
O = P @ V
```

**Issues**:
1. **Memory**: Stores S and P matrices → O(N²) memory for sequence length N
2. **HBM Bandwidth**: Multiple slow HBM reads/writes
3. **Long Sequences**: 16K tokens needs 1GB just for attention matrices!

### Memory Hierarchy

```
Registers (RF)    ~20KB   | Fastest, tiny
Shared Memory     ~150KB  | Fast, small (SRAM)
L2 Cache          ~40MB   | Medium speed
HBM (Global Mem)  ~40GB   | Slow, large
```

**Problem**: Standard attention constantly reads/writes to slow HBM.

---

## Key Innovation: IO-Aware Algorithm

### Core Idea

Instead of:
1. Compute full S = QK^T (write to HBM)
2. Compute full P = softmax(S) (read S, write P to HBM)
3. Compute O = PV (read P, write O)

Do:
1. **Tile** Q, K, V into blocks that fit in SRAM
2. Compute attention **incrementally** in SRAM
3. **Never materialize** full S or P in HBM
4. **Recompute** in backward pass (trade compute for memory)

---

## Algorithm Breakdown

### Forward Pass

```
Algorithm: Flash Attention Forward

Input: Q, K, V ∈ R^(N×d)  # On HBM
Output: O ∈ R^(N×d), L ∈ R^N  # Output and softmax normalizer

# Divide into blocks
Divide Q into T_r blocks Q_1, ..., Q_{T_r} of size B_r each
Divide K, V into T_c blocks K_1, V_1, ..., K_{T_c}, V_{T_c} of size B_c each

# Initialize output and statistics
Initialize O = (0)_{N×d} in HBM
Initialize l = (0)_N, m = (-∞)_N in HBM  # normalizer and max

# Process each Q block
For i = 1, ..., T_r:
    # Load Q block to SRAM
    Load Q_i from HBM to SRAM
    Load O_i, l_i, m_i from HBM to SRAM

    # Process each K,V block
    For j = 1, ..., T_c:
        # Load K,V block to SRAM
        Load K_j, V_j from HBM to SRAM

        # Compute attention for this block pair
        S_ij = Q_i @ K_j^T  # In SRAM

        # Online softmax: Update running max and sum
        m_ij^new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_ij^new)
        l_ij^new = exp(m_i - m_ij^new) * l_i + rowsum(P_ij)

        # Update output with new contribution
        O_i = (l_i * exp(m_i - m_ij^new) * O_i + P_ij @ V_j) / l_ij^new

        # Update statistics
        m_i = m_ij^new
        l_i = l_ij^new

    # Write back to HBM
    Write O_i, l_i, m_i to HBM
```

### Key Techniques

#### 1. **Tiling**
- Split Q, K, V into blocks: B_r × d and B_c × d
- Each block fits in SRAM (~150KB)
- Compute attention block-by-block

#### 2. **Online Softmax**
- Compute softmax incrementally without storing full QK^T
- Track running max (m) and sum (l)
- Update formula:
  ```
  softmax([x, y]) = [exp(x - m)·l_x + exp(y - m)·l_y] / (l_x + l_y)
  where m = max(max(x), max(y))
  ```

#### 3. **Recomputation in Backward**
- Don't store S or P for backward pass
- Recompute them from Q, K in backward
- Saves O(N²) memory at cost of extra compute

---

## Implementation Details

### Block Size Selection

**Constraints**:
- All of Q_i, K_j, V_j, S_ij, P_ij must fit in SRAM
- SRAM size ≈ 100-200 KB

**Trade-offs**:
- Larger blocks → Fewer HBM accesses (better)
- Larger blocks → May not fit in SRAM (worse)

**Optimal**: B_c = ceil(M / (4d)), B_r = min(ceil(M / (4d)), d)
where M = SRAM size in bytes, d = head dimension

**Example** (A100 with 192KB SRAM, d=64, FP16):
- M = 192 * 1024 bytes
- B_c = ceil(192KB / (4 * 64 * 2)) = ceil(192KB / 512B) ≈ 384
- B_r = min(384, 64) = 64

### CUDA Implementation Highlights

```cuda
// Pseudocode for Flash Attention kernel

__global__ void flash_attention_forward(
    const half* Q,  // [batch, heads, seqlen, headdim]
    const half* K,
    const half* V,
    half* O,
    int seqlen,
    int headdim
) {
    // Shared memory for blocks
    __shared__ half Qi[Br][d];    // Q block
    __shared__ half Kj[Bc][d];    // K block
    __shared__ half Vj[Bc][d];    // V block
    __shared__ half Sij[Br][Bc];  // Attention scores

    // Registers for output and statistics
    float Oi[d];  // Output accumulator
    float li, mi;  // Softmax statistics

    // Initialize
    mi = -INFINITY;
    li = 0.0f;
    for (int d = 0; d < headdim; d++) Oi[d] = 0.0f;

    // Load Q block (cooperatively)
    load_block_sram(Q, Qi, ...);

    // Loop over K,V blocks
    for (int j = 0; j < Tc; j++) {
        // Load K, V blocks
        load_block_sram(K + j*Bc*d, Kj, ...);
        load_block_sram(V + j*Bc*d, Vj, ...);

        // Compute S_ij = Q_i @ K_j^T
        matmul_shared(Qi, Kj, Sij, ...);

        // Online softmax update
        float mij_new = max(mi, rowmax(Sij));
        float scale_old = exp(mi - mij_new);

        // Compute P_ij = exp(S_ij - mij_new)
        for (int r = 0; r < Br; r++) {
            for (int c = 0; c < Bc; c++) {
                Sij[r][c] = exp(Sij[r][c] - mij_new);
            }
        }

        // Update sum
        float lij_new = scale_old * li + rowsum(Sij);

        // Update output: O_i = (scale_old * li * O_i + P_ij @ V_j) / lij_new
        float scale_factor = scale_old * li / lij_new;
        for (int d = 0; d < headdim; d++) {
            Oi[d] *= scale_factor;
        }

        // Add contribution from P_ij @ V_j
        matmul_accumulate(Sij, Vj, Oi, 1.0f / lij_new, ...);

        // Update statistics
        mi = mij_new;
        li = lij_new;
    }

    // Write output to HBM
    store_output(O, Oi, ...);
}
```

---

## Performance Analysis

### Memory Complexity

**Standard Attention**:
- Forward: O(N²) memory for S and P
- Backward: O(N²) memory for storing gradients

**Flash Attention**:
- Forward: O(N) memory (only Q, K, V, O)
- Backward: O(N) memory (recompute S, P)

**Savings**: From O(N²) to O(N) → **Enables 4x longer sequences**

### Computational Complexity

Both algorithms: O(N² d) FLOPs

**Flash Attention trades**:
- More FLOPs (recomputation in backward)
- For fewer HBM accesses

**Why faster?** Memory bandwidth is the bottleneck, not compute!

### HBM Accesses

**Standard Attention** (N×d inputs, d=64):
- Load Q, K, V: 3Nd
- Store S: N²  ← **Expensive!**
- Load S for softmax: N²
- Store P: N²  ← **Expensive!**
- Load P for PV: N²
- Store O: Nd
- **Total**: O(N² + Nd) ≈ O(N²) for large N

**Flash Attention** (with blocks of size B):
- Outer loop (Tr iterations): Load/store O, l, m per Q block
- Inner loop (Tc iterations per Tr): Load K, V blocks
- No S or P stored to HBM!
- **Total**: O(N²d² / M) where M = SRAM size
- **Practical**: ~10-20x fewer HBM accesses

### Benchmark Results (A100 GPU)

| Sequence Length | Standard Attn | Flash Attn | Speedup |
|----------------|---------------|------------|---------|
| 512            | 0.15 ms       | 0.08 ms    | 1.9x    |
| 1024           | 0.52 ms       | 0.22 ms    | 2.4x    |
| 2048           | 2.01 ms       | 0.72 ms    | 2.8x    |
| 4096           | 8.15 ms       | 2.51 ms    | 3.2x    |
| 8192           | OOM           | 9.53 ms    | ∞       |

**Memory Usage** (sequence length 2048, batch 8, 12 heads, d=64):
- Standard: ~2.1 GB
- Flash Attention: ~0.5 GB (**4.2x reduction**)

---

## When to Use Flash Attention

### ✅ Use Flash Attention When:

1. **Long sequences** (N > 1024)
   - Standard attention runs out of memory
   - Flash Attention enables much longer contexts

2. **Training large models**
   - Memory savings allow larger batches
   - Faster training overall

3. **Memory is the bottleneck**
   - A100, H100 GPUs with high compute-to-memory ratio
   - Flash Attention better utilizes hardware

4. **Exact attention needed**
   - Flash Attention is mathematically exact
   - No approximation errors

### ❌ Don't Use Flash Attention When:

1. **Very short sequences** (N < 256)
   - Overhead of tiling may not pay off
   - Standard attention is simpler

2. **Sparse attention patterns needed**
   - Flash Attention computes dense attention
   - Use specialized sparse kernels instead

3. **Older GPUs** (pre-Volta)
   - Benefits are smaller
   - May not have enough SRAM

---

## Practical Applications

### 1. Language Models

```python
# PyTorch integration
from flash_attn import flash_attn_func

def forward(self, x):
    # x: [batch, seqlen, hidden_dim]
    qkv = self.qkv_proj(x)  # [batch, seqlen, 3*hidden_dim]
    q, k, v = qkv.chunk(3, dim=-1)

    # Reshape for multi-head
    q = q.view(batch, seqlen, num_heads, head_dim)
    k = k.view(batch, seqlen, num_heads, head_dim)
    v = v.view(batch, seqlen, num_heads, head_dim)

    # Flash Attention (drop-in replacement)
    out = flash_attn_func(q, k, v, causal=True)

    return self.out_proj(out)
```

### 2. Training Speedup

GPT-2 style model training:
- **Standard attention**: 3.2 hours/epoch, max batch size 32
- **Flash Attention**: 2.1 hours/epoch, max batch size 64
- **Result**: 2.4x faster training (1.5x from speed + 1.6x from batch)

### 3. Enabling Longer Contexts

- Standard: Max 2048 tokens on 40GB A100
- Flash Attention: **8192 tokens** on same hardware
- Critical for: Document Q&A, code completion, long-form generation

---

## Extensions and Follow-ups

### Flash Attention 2 (2023)

Improvements over V1:
- Better parallelization (split-Q instead of split-KV)
- 1.5-2x faster than Flash Attention 1
- Simpler implementation

### Flash Attention 3 (2024)

Hardware-specific optimizations for Hopper (H100):
- Uses new Tensor Memory Accelerator (TMA)
- Async warp specialization
- Up to 1.5x faster than FA2 on H100

### Related Work

- **Self-attention Does Not Need O(n²) Memory** (Rabe & Staats, 2021): Theoretical foundation
- **Flash-Decoding**: Optimizations for autoregressive decoding
- **Paged Attention** (vLLM): KV cache management for inference

---

## Implementation in LeetCUDA

### Code Structure

```
kernels/flash-attn/
├── flash_attn_v1.cu          # Split-KV version
├── flash_attn_v2.cu          # Split-Q version (faster)
├── flash_attn_forward.cu     # Forward pass
├── flash_attn_backward.cu    # Backward pass
├── flash_attn.py             # PyTorch bindings
├── test_flash_attn.py        # Correctness tests
└── benchmark_flash_attn.py   # Performance benchmarks
```

### Usage Example

```python
import torch
from flash_attn import flash_attention_forward

# Create random Q, K, V
batch, seqlen, nheads, headdim = 4, 2048, 12, 64
q = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)

# Flash Attention
output = flash_attention_forward(q, k, v, causal=False)

# Verify correctness against PyTorch
output_ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2),
    k.transpose(1, 2),
    v.transpose(1, 2)
).transpose(1, 2)

print(f"Max error: {(output - output_ref).abs().max().item()}")  # Should be ~1e-3
```

---

## Learning Path

### Prerequisites

Before studying Flash Attention:
1. **L2: Softmax** - Understand numerical stability
2. **L3: Attention Basics** - Know standard attention algorithm
3. **L3: GEMM** - Understand matrix multiplication optimization
4. **L2: Tiling** - Grasp tiling strategies

### Tutorial Progression

1. `tutorials/advanced/attention-basics.md` - Standard attention
2. `tutorials/expert/online-softmax.md` - Incremental softmax
3. `tutorials/expert/flash-attention.md` - Complete Flash Attention
4. `projects/expert/flash-attention-library/` - Full implementation

### Estimated Learning Time

- **Reading paper**: 2-3 hours
- **Understanding algorithm**: 4-6 hours
- **Implementation**: 15-20 hours
- **Optimization**: 10-15 hours
- **Total**: ~40 hours for expert implementation

---

## Key Takeaways

1. **IO-awareness is critical**: Optimizing memory accesses matters more than FLOPs
2. **Recomputation can be cheap**: Trading compute for memory bandwidth is often beneficial
3. **Tiling is powerful**: Blocking to fit SRAM enables huge optimizations
4. **Online algorithms**: Computing incrementally (online softmax) avoids materialization
5. **Hardware matters**: Understanding memory hierarchy guides algorithm design

---

## References

### Original Paper
```bibtex
@inproceedings{dao2022flashattention,
  title={Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

### Further Reading
- [Flash Attention GitHub](https://github.com/HazyResearch/flash-attention)
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- [Blog: Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [CUDA Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**Generated by**: Content Creator Agent #3
**Session**: session_20251119_051648
**Last Updated**: 2025-11-19
**Next**: Read [Flash Attention 2](./flash-attention-2.md) for improvements
