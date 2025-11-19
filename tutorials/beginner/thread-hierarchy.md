# Tutorial: Mastering CUDA Thread Hierarchy

## Learning Objectives

By the end of this tutorial, you will:
- ✅ Calculate global thread IDs for 1D, 2D, and 3D configurations
- ✅ Choose optimal grid and block sizes for your problem
- ✅ Understand thread, block, and grid organization
- ✅ Handle boundary conditions correctly
- ✅ Optimize thread configurations for performance

**Level**: L1 (Beginner)
**Time**: 60-75 minutes
**Prerequisites**: [Your First CUDA Kernel](./your-first-cuda-kernel.md)

---

## Introduction

### The Thread Hierarchy

CUDA organizes threads in a **3-level hierarchy**:

```
Grid (All threads for a kernel launch)
  └── Blocks (Groups of threads)
        └── Threads (Individual execution units)
```

**Why This Hierarchy?**
1. **Scalability**: Works on GPUs with different numbers of SMs
2. **Flexibility**: Handle 1D, 2D, 3D problems naturally
3. **Optimization**: Shared memory and synchronization at block level

---

## Part 1: Understanding the Hierarchy

### Level 1: Thread

**The smallest unit of execution**

```cuda
__global__ void kernel() {
    // Each thread has a unique ID within its block
    int tx = threadIdx.x;  // 0, 1, 2, ..., blockDim.x-1
    int ty = threadIdx.y;  // (if using 2D blocks)
    int tz = threadIdx.z;  // (if using 3D blocks)

    printf("Thread (%d, %d, %d)\n", tx, ty, tz);
}
```

**Key Properties**:
- Has unique ID within block (`threadIdx`)
- Executes kernel code
- Can access shared memory within its block
- Typical block size: 128-1024 threads

---

### Level 2: Block

**Group of threads that can cooperate**

```cuda
__global__ void kernel() {
    // Each block has a unique ID within the grid
    int bx = blockIdx.x;   // Block's X position in grid
    int by = blockIdx.y;   // Block's Y position in grid
    int bz = blockIdx.z;   // Block's Z position in grid

    // Block dimensions
    int block_size_x = blockDim.x;  // Threads per block in X
    int block_size_y = blockDim.y;  // Threads per block in Y
    int block_size_z = blockDim.z;  // Threads per block in Z

    printf("Block (%d, %d, %d) with %d threads\n",
           bx, by, bz, block_size_x * block_size_y * block_size_z);
}
```

**Key Properties**:
- Has unique ID within grid (`blockIdx`)
- Contains multiple threads (up to 1024 typically)
- Threads in same block can:
  - Share data via shared memory
  - Synchronize with `__syncthreads()`
- Blocks execute independently!

---

### Level 3: Grid

**Collection of all blocks for a kernel launch**

```cuda
// Host code
dim3 gridDim(num_blocks_x, num_blocks_y, num_blocks_z);
dim3 blockDim(threads_per_block_x, threads_per_block_y, threads_per_block_z);

kernel<<<gridDim, blockDim>>>(args...);
```

**Key Properties**:
- Defines total amount of parallelism
- Grid dimensions (`gridDim.x`, `gridDim.y`, `gridDim.z`)
- All blocks in grid run the same kernel
- Blocks may execute in any order!

---

## Part 2: 1D Thread Indexing

### Simple 1D Problem: Vector Addition

**Problem**: Add two vectors of size N

```cuda
// Vector: [a0, a1, a2, a3, ..., aN-1]
//       + [b0, b1, b2, b3, ..., bN-1]
//       = [c0, c1, c2, c3, ..., cN-1]
```

### Kernel Code

```cuda
__global__ void vectorAdd(float* a, float* b, float* c, int N) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
```

### Thread ID Calculation Breakdown

```
Launch: <<<4, 256>>>  (4 blocks, 256 threads each)

Block 0:
  Thread 0: i = 0 * 256 + 0 = 0
  Thread 1: i = 0 * 256 + 1 = 1
  ...
  Thread 255: i = 0 * 256 + 255 = 255

Block 1:
  Thread 0: i = 1 * 256 + 0 = 256
  Thread 1: i = 1 * 256 + 1 = 257
  ...
  Thread 255: i = 1 * 256 + 255 = 511

Block 2:
  Thread 0: i = 2 * 256 + 0 = 512
  ...

Block 3:
  Thread 0: i = 3 * 256 + 0 = 768
  ...
  Thread 255: i = 3 * 256 + 255 = 1023
```

**Formula**: `i = blockIdx.x * blockDim.x + threadIdx.x`

---

### Choosing Grid and Block Sizes

**For N = 1,000,000 elements**:

**Option 1**: 256 threads per block
```cpp
int threads = 256;
int blocks = (N + threads - 1) / threads;  // Ceiling division
// blocks = (1000000 + 255) / 256 = 3907 blocks

vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
```

**Option 2**: 512 threads per block
```cpp
int threads = 512;
int blocks = (N + threads - 1) / threads;
// blocks = (1000000 + 511) / 512 = 1954 blocks

vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
```

**Why ceiling division?**
```
N = 1000, threads = 256

Wrong: blocks = 1000 / 256 = 3 (integer division)
       Total threads = 3 × 256 = 768 (not enough!)

Right: blocks = (1000 + 255) / 256 = 4
       Total threads = 4 × 256 = 1024 (covers all elements)
```

---

## Part 3: 2D Thread Indexing

### 2D Problem: Matrix Addition

**Problem**: Add two N×M matrices

```
Matrix A (3×4):
┌─────────────────┐
│ a00 a01 a02 a03 │
│ a10 a11 a12 a13 │
│ a20 a21 a22 a23 │
└─────────────────┘

Want to compute: C[row][col] = A[row][col] + B[row][col]
```

### Kernel Code

```cuda
__global__ void matrixAdd(float* A, float* B, float* C,
                         int rows, int cols) {
    // Calculate 2D indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < rows && col < cols) {
        // Convert 2D indices to 1D array index
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
```

### Launch Configuration

```cpp
// Matrix size: 4096 × 2048
int rows = 4096;
int cols = 2048;

// Block size: 16×16 threads (256 total threads per block)
dim3 blockDim(16, 16);

// Grid size: Calculate how many blocks needed
dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,  // X direction
             (rows + blockDim.y - 1) / blockDim.y); // Y direction

// gridDim = (128, 256)  → 128 blocks in X, 256 blocks in Y

matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
```

### Visual Representation

```
Grid (128 × 256 blocks)
┌────────────────────────────────────────┐
│ Block(0,0)  Block(1,0)  ... Block(127,0)│
│  16×16       16×16          16×16       │
│                                         │
│ Block(0,1)  Block(1,1)  ... Block(127,1)│
│  16×16       16×16          16×16       │
│                                         │
│    ...        ...      ...     ...      │
│                                         │
│ Block(0,255) ...        ... Block(127,255)│
│  16×16                      16×16       │
└────────────────────────────────────────┘

Each block (e.g., Block(0,0)):
┌─────────────────────────────┐
│ T(0,0) T(1,0) ... T(15,0)   │
│ T(0,1) T(1,1) ... T(15,1)   │
│  ...    ...   ...   ...     │
│ T(0,15) T(1,15)... T(15,15) │
└─────────────────────────────┘
```

### Thread Index Calculation

**For a thread in Block(2, 3), Thread(5, 7)**:

```cuda
row = blockIdx.y * blockDim.y + threadIdx.y
    = 3 * 16 + 7
    = 55

col = blockIdx.x * blockDim.x + threadIdx.x
    = 2 * 16 + 5
    = 37

idx = row * cols + col
    = 55 * 2048 + 37
    = 112,677
```

This thread processes element at row 55, column 37 of the matrix!

---

## Part 4: 3D Thread Indexing

### 3D Problem: Volume Processing

**Problem**: Process a 3D medical image (CT scan)

```cpp
// Volume dimensions
int depth = 256;   // Z
int height = 256;  // Y
int width = 256;   // X
```

### Kernel Code

```cuda
__global__ void processVolume(float* input, float* output,
                             int width, int height, int depth) {
    // Calculate 3D indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check
    if (x < width && y < height && z < depth) {
        // Convert 3D indices to 1D array index
        int idx = z * (height * width) + y * width + x;

        // Process voxel
        output[idx] = processVoxel(input[idx]);
    }
}
```

### Launch Configuration

```cpp
// Block: 8×8×8 threads (512 total threads per block)
dim3 blockDim(8, 8, 8);

// Grid: Calculate blocks needed
dim3 gridDim((width + blockDim.x - 1) / blockDim.x,   // X: 32 blocks
             (height + blockDim.y - 1) / blockDim.y,  // Y: 32 blocks
             (depth + blockDim.z - 1) / blockDim.z);  // Z: 32 blocks

// Total: 32×32×32 = 32,768 blocks
// Total threads: 32,768 × 512 = 16,777,216 threads!

processVolume<<<gridDim, blockDim>>>(d_input, d_output, width, height, depth);
```

### 3D Index Calculation

**Formulas**:
```cuda
x = blockIdx.x * blockDim.x + threadIdx.x
y = blockIdx.y * blockDim.y + threadIdx.y
z = blockIdx.z * blockDim.z + threadIdx.z

// 3D to 1D conversion (row-major order)
idx = z * (height * width) + y * width + x
```

**Example**: Thread in Block(4,5,6), Thread(3,2,1)
```cuda
x = 4 * 8 + 3 = 35
y = 5 * 8 + 2 = 42
z = 6 * 8 + 1 = 49

idx = 49 * (256 * 256) + 42 * 256 + 35
    = 3,211,264 + 10,752 + 35
    = 3,222,051
```

---

## Part 5: Choosing Optimal Block Sizes

### Constraints

**Hardware Limits** (example: RTX 3080):
- Max threads per block: 1024
- Max block dimensions: (1024, 1024, 64)
- Max grid dimensions: (2³¹-1, 65535, 65535)

**Best Practices**:

#### 1. **Multiple of Warp Size (32)**

```cpp
// GOOD: Multiples of 32
int threads = 128;  // 4 warps
int threads = 256;  // 8 warps
int threads = 512;  // 16 warps

// BAD: Not multiples of 32
int threads = 100;  // Wastes 28 threads in 4th warp
int threads = 500;  // Wastes 12 threads in 16th warp
```

#### 2. **Balance Occupancy and Resources**

**Occupancy**: Percentage of GPU actively executing threads

```cpp
// Option A: 128 threads/block
// → More blocks, better occupancy
// → Less shared memory per thread

// Option B: 1024 threads/block
// → Fewer blocks, might limit occupancy
// → More shared memory per thread

// Sweet spot often: 256 or 512 threads/block
```

#### 3. **Match Problem Dimensions**

**For 2D problems**:
```cpp
// Image processing: 16×16 = 256 threads (GOOD)
dim3 blockDim(16, 16);

// Matrix operations: 32×32 = 1024 threads (GOOD)
dim3 blockDim(32, 32);

// Avoid non-square for 2D (unless problem requires it)
dim3 blockDim(64, 8);  // OK, but less cache-friendly
```

**For 3D problems**:
```cpp
// 8×8×8 = 512 threads (GOOD for volumes)
dim3 blockDim(8, 8, 8);

// 4×4×4 = 64 threads (Too small, low occupancy)
// 16×16×16 = 4096 threads (Too large, exceeds limit!)
```

---

## Part 6: Handling Boundary Conditions

### Problem: N not divisible by block size

**Scenario**: N = 1000, threads = 256

```
Total threads launched: 4 × 256 = 1024
Valid indices: 0-999 (1000 elements)
Invalid indices: 1000-1023 (24 extra threads)
```

### Solution 1: Simple Boundary Check

```cuda
__global__ void kernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {  // ← Critical!
        data[i] = process(data[i]);
    }
    // Threads with i >= N do nothing
}
```

### Solution 2: Early Return

```cuda
__global__ void kernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;  // Early exit

    // Rest of kernel
    data[i] = process(data[i]);
}
```

### Solution 3: Grid-Stride Loop (for large N)

```cuda
__global__ void kernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes multiple elements
    for (; i < N; i += stride) {
        data[i] = process(data[i]);
    }
}

// Benefit: Works for any N, even if N >> grid size
```

---

## Part 7: Common Patterns

### Pattern 1: 1D Array Processing

```cuda
// Problem: Process 1D array
__global__ void process1D(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = transform(data[i]);
    }
}

// Launch
int threads = 256;
int blocks = (N + threads - 1) / threads;
process1D<<<blocks, threads>>>(d_data, N);
```

### Pattern 2: 2D Matrix Processing

```cuda
// Problem: Process 2D matrix
__global__ void process2D(float* data, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        data[idx] = transform(data[idx]);
    }
}

// Launch
dim3 threads(16, 16);
dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
process2D<<<blocks, threads>>>(d_data, rows, cols);
```

### Pattern 3: Strided Access

```cuda
// Problem: Process every N-th element
__global__ void processStrided(float* data, int N, int stride) {
    int base = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = base; i < N; i += stride * gridDim.x * blockDim.x) {
        data[i] = transform(data[i]);
    }
}
```

---

## Practice Exercises

### Exercise 1: Calculate Thread IDs

Given:
- Block size: 512 threads
- Grid size: 256 blocks
- Thread in Block 10, Thread 327

Calculate the global thread ID.

<details>
<summary>Solution</summary>

```cuda
i = blockIdx.x * blockDim.x + threadIdx.x
  = 10 * 512 + 327
  = 5447
```
</details>

### Exercise 2: Grid Configuration

Design a grid/block configuration for:
- Problem: 2048 × 2048 matrix
- Constraint: Use 16×16 thread blocks

<details>
<summary>Solution</summary>

```cpp
dim3 blockDim(16, 16);  // 256 threads per block
dim3 gridDim((2048 + 15) / 16,  // X: 128 blocks
             (2048 + 15) / 16);  // Y: 128 blocks

// Total: 128 × 128 = 16,384 blocks
// Total threads: 16,384 × 256 = 4,194,304 threads
// Covers 2048 × 2048 = 4,194,304 elements perfectly!
```
</details>

### Exercise 3: Implement 2D Transpose

Write a kernel to transpose a matrix (swap rows and columns).

<details>
<summary>Solution</summary>

```cuda
__global__ void transpose(float* input, float* output,
                         int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int input_idx = row * cols + col;
        int output_idx = col * rows + row;  // Swapped!
        output[output_idx] = input[input_idx];
    }
}

// Launch
dim3 threads(16, 16);
dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
transpose<<<blocks, threads>>>(d_in, d_out, rows, cols);
```
</details>

---

## Summary

### Key Formulas

**1D**:
```cuda
i = blockIdx.x * blockDim.x + threadIdx.x;
```

**2D**:
```cuda
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
idx = row * width + col;
```

**3D**:
```cuda
x = blockIdx.x * blockDim.x + threadIdx.x;
y = blockIdx.y * blockDim.y + threadIdx.y;
z = blockIdx.z * blockDim.z + threadIdx.z;
idx = z * (height * width) + y * width + x;
```

### Best Practices

✅ **Always** include boundary checks: `if (i < N)`
✅ **Use** multiples of 32 for thread counts
✅ **Prefer** 256-512 threads per block (sweet spot)
✅ **Match** block dimensions to problem (2D for matrices, etc.)
✅ **Calculate** grid size with ceiling division: `(N + threads - 1) / threads`

---

## Next Steps

1. ✅ **Practice**: Implement exercises above
2. ✅ **Next Tutorial**: [Memory Model Basics](./memory-basics.md)
3. ✅ **Project**: [Activation Library](../../projects/beginner/activation-library/)

---

**Tutorial Generated by**: Content Creator Agent #4
**Difficulty**: Beginner (L1)
**Time**: 60-75 minutes
