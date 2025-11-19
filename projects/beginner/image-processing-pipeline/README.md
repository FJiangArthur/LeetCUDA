# Project: CUDA Image Processing Pipeline

## Project Overview

Build a complete image processing library with GPU-accelerated filters and transformations, learning fundamental CUDA patterns while creating a practical tool for real-world image manipulation.

**Difficulty**: ⭐⭐ Beginner
**Time Estimate**: 10-15 hours
**Level**: L1 (Basic Kernels)

---

## Learning Objectives

By completing this project, you will:

- ✅ Implement 8+ image processing kernels (filters, transformations, effects)
- ✅ Work with 2D data structures and multi-channel images
- ✅ Optimize memory access patterns for spatial data
- ✅ Handle image formats (RGB, grayscale, alpha channels)
- ✅ Create reusable library with Python bindings
- ✅ Benchmark against CPU implementations (expect 20-50x speedups!)

**Prerequisites**:
- Your First CUDA Kernel
- Thread Hierarchy (2D grids)
- Memory Model Basics
- Element-wise Operations

---

## Project Structure

```
image-processing-pipeline/
├── README.md                    # This file
├── src/
│   ├── kernels/
│   │   ├── grayscale.cu         # RGB → Grayscale conversion
│   │   ├── blur.cu              # Gaussian blur
│   │   ├── sharpen.cu           # Sharpening filter
│   │   ├── edge_detect.cu       # Sobel edge detection
│   │   ├── brightness.cu        # Brightness/contrast adjustment
│   │   ├── flip.cu              # Horizontal/vertical flip
│   │   ├── rotate.cu            # 90°/180°/270° rotation
│   │   └── sepia.cu             # Sepia tone effect
│   ├── image_utils.cu           # Image loading/saving
│   └── pipeline.cu              # Main pipeline orchestration
├── include/
│   └── image_processing.h       # Public API
├── python/
│   ├── setup.py                 # PyTorch extension build
│   └── image_cuda.cpp           # Python bindings
├── tests/
│   ├── test_kernels.cu          # Correctness tests
│   └── benchmark.cu             # Performance benchmarks
├── examples/
│   ├── basic_filters.cu         # Simple usage examples
│   └── batch_processing.cu      # Batch image processing
└── images/
    ├── test/                    # Test images
    └── output/                  # Results directory
```

---

## Phase 1: Foundation (3-4 hours)

### Task 1.1: Image Data Structure

Create basic image representation:

```cuda
// include/image_processing.h

struct Image {
    unsigned char* data;  // Raw pixel data (HWC layout)
    int width;
    int height;
    int channels;         // 1=grayscale, 3=RGB, 4=RGBA
    size_t pitch;         // For aligned memory
};

// Allocation helper
Image* allocate_image(int width, int height, int channels);
void free_image(Image* img);

// Host ↔ Device transfers
Image* image_to_device(const Image* h_img);
Image* image_to_host(const Image* d_img);
```

**Key decision**: Use **HWC layout** (Height × Width × Channels) for coalesced access:
```
Memory: [R₀ G₀ B₀ R₁ G₁ B₁ R₂ G₂ B₂ ...]
        ↑ Each pixel's channels together
```

**Why not CHW** (Channels × Height × Width)?
- HWC: Better coalescing when processing all channels together
- CHW: Better for channel-specific operations (less common in image processing)

### Task 1.2: Image I/O

Implement loading/saving (use STB library for simplicity):

```cuda
// src/image_utils.cu

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image* load_image(const char* filename) {
    Image* img = (Image*)malloc(sizeof(Image));

    img->data = stbi_load(filename, &img->width, &img->height, &img->channels, 0);

    if (!img->data) {
        fprintf(stderr, "Error loading image: %s\n", filename);
        free(img);
        return nullptr;
    }

    return img;
}

void save_image(const char* filename, const Image* img) {
    stbi_write_png(filename, img->width, img->height, img->channels,
                   img->data, img->width * img->channels);
}
```

**Testing**: Load and save test image to verify pipeline works.

---

## Phase 2: Basic Filters (4-5 hours)

### Kernel 2.1: RGB to Grayscale

**Formula**: `Gray = 0.299*R + 0.587*G + 0.114*B` (perceptual luminance)

```cuda
// src/kernels/grayscale.cu

__global__ void rgb_to_gray_kernel(
    const unsigned char* rgb,
    unsigned char* gray,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;  // RGB index

        unsigned char r = rgb[idx + 0];
        unsigned char g = rgb[idx + 1];
        unsigned char b = rgb[idx + 2];

        // Weighted average for perceptual accuracy
        gray[y * width + x] = (unsigned char)(
            0.299f * r + 0.587f * g + 0.114f * b
        );
    }
}
```

**Launch configuration**:
```cuda
dim3 block(16, 16);  // 256 threads per block
dim3 grid((width + 15) / 16, (height + 15) / 16);

rgb_to_gray_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
```

**Expected performance**: 500-700 GB/s (memory-bound, near peak)

### Kernel 2.2: Brightness and Contrast

**Formula**: `output = clamp(alpha * input + beta, 0, 255)`
- `alpha > 1`: Increase contrast
- `beta > 0`: Increase brightness

```cuda
__global__ void brightness_contrast_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    float alpha,  // Contrast (typically 0.5 - 3.0)
    float beta    // Brightness (typically -100 to +100)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        for (int c = 0; c < channels; c++) {
            float value = alpha * input[idx + c] + beta;
            output[idx + c] = (unsigned char)fminf(fmaxf(value, 0.0f), 255.0f);
        }
    }
}
```

**Optimization opportunity**: Loop is small → compiler will unroll for channels=3.

### Kernel 2.3: Sepia Tone

**Sepia matrix transformation**:
```
R_out = 0.393*R + 0.769*G + 0.189*B
G_out = 0.349*R + 0.686*G + 0.168*B
B_out = 0.272*R + 0.534*G + 0.131*B
```

```cuda
__global__ void sepia_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        float r = input[idx + 0];
        float g = input[idx + 1];
        float b = input[idx + 2];

        // Sepia transformation
        float tr = 0.393f * r + 0.769f * g + 0.189f * b;
        float tg = 0.349f * r + 0.686f * g + 0.168f * b;
        float tb = 0.272f * r + 0.534f * g + 0.131f * b;

        output[idx + 0] = (unsigned char)fminf(tr, 255.0f);
        output[idx + 1] = (unsigned char)fminf(tg, 255.0f);
        output[idx + 2] = (unsigned char)fminf(tb, 255.0f);
    }
}
```

**Exercise**: Try other color transformations (invert, posterize, etc.)

---

## Phase 3: Spatial Filters (4-5 hours)

### Kernel 3.1: Gaussian Blur

**Concept**: Average pixels with weighted neighbors (smooth image)

**3×3 Gaussian kernel**:
```
1/16 * | 1  2  1 |
       | 2  4  2 |
       | 1  2  1 |
```

```cuda
__global__ void gaussian_blur_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Gaussian 3x3 kernel weights
        const float kernel[9] = {
            1.0f/16, 2.0f/16, 1.0f/16,
            2.0f/16, 4.0f/16, 2.0f/16,
            1.0f/16, 2.0f/16, 1.0f/16
        };

        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;

            // Apply kernel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    int idx = (py * width + px) * channels + c;

                    sum += kernel[(ky+1)*3 + (kx+1)] * input[idx];
                }
            }

            output[(y * width + x) * channels + c] = (unsigned char)sum;
        }
    }
}
```

**Optimization ideas**:
- Use shared memory to cache neighborhood (avoid redundant global reads)
- Separate horizontal and vertical passes (separable filter)
- Use constant memory for kernel weights

### Kernel 3.2: Sharpen Filter

**Unsharp masking**: `output = original + alpha * (original - blurred)`

Or use sharpening kernel directly:
```
| 0  -1   0 |
|-1   5  -1 |
| 0  -1   0 |
```

```cuda
__global__ void sharpen_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        for (int c = 0; c < channels; c++) {
            int idx = (y * width + x) * channels + c;

            float center = 5.0f * input[idx];
            float top    = input[((y-1) * width + x) * channels + c];
            float bottom = input[((y+1) * width + x) * channels + c];
            float left   = input[(y * width + (x-1)) * channels + c];
            float right  = input[(y * width + (x+1)) * channels + c];

            float result = center - top - bottom - left - right;

            output[idx] = (unsigned char)fminf(fmaxf(result, 0.0f), 255.0f);
        }
    }
}
```

### Kernel 3.3: Edge Detection (Sobel)

**Sobel operator**: Detects edges using gradient

**Horizontal (Gx)**:
```
|-1  0  1|
|-2  0  2|
|-1  0  1|
```

**Vertical (Gy)**:
```
|-1 -2 -1|
| 0  0  0|
| 1  2  1|
```

**Magnitude**: `sqrt(Gx² + Gy²)`

```cuda
__global__ void sobel_edge_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel operators
        float gx = 0.0f, gy = 0.0f;

        // Horizontal gradient (Gx)
        gx += -1.0f * input[(y-1) * width + (x-1)];
        gx +=  1.0f * input[(y-1) * width + (x+1)];
        gx += -2.0f * input[y * width + (x-1)];
        gx +=  2.0f * input[y * width + (x+1)];
        gx += -1.0f * input[(y+1) * width + (x-1)];
        gx +=  1.0f * input[(y+1) * width + (x+1)];

        // Vertical gradient (Gy)
        gy += -1.0f * input[(y-1) * width + (x-1)];
        gy += -2.0f * input[(y-1) * width + x];
        gy += -1.0f * input[(y-1) * width + (x+1)];
        gy +=  1.0f * input[(y+1) * width + (x-1)];
        gy +=  2.0f * input[(y+1) * width + x];
        gy +=  1.0f * input[(y+1) * width + (x+1)];

        // Gradient magnitude
        float magnitude = sqrtf(gx * gx + gy * gy);

        output[y * width + x] = (unsigned char)fminf(magnitude, 255.0f);
    }
}
```

**Note**: Input should be grayscale for edge detection.

---

## Phase 4: Geometric Transformations (3-4 hours)

### Kernel 4.1: Image Flip

**Horizontal flip**: Mirror left-right
**Vertical flip**: Mirror top-bottom

```cuda
__global__ void flip_horizontal_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_idx = (y * width + x) * channels;
        int out_idx = (y * width + (width - 1 - x)) * channels;

        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = input[in_idx + c];
        }
    }
}

__global__ void flip_vertical_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_idx = (y * width + x) * channels;
        int out_idx = ((height - 1 - y) * width + x) * channels;

        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = input[in_idx + c];
        }
    }
}
```

### Kernel 4.2: 90° Rotation

**Right rotation** (clockwise):
```
(x, y) → (height - 1 - y, x)
```

```cuda
__global__ void rotate_90_cw_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_idx = (y * width + x) * channels;
        // After rotation: new width = old height, new height = old width
        int new_x = height - 1 - y;
        int new_y = x;
        int out_idx = (new_y * height + new_x) * channels;

        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = input[in_idx + c];
        }
    }
}
```

**Challenge**: Handle output image dimensions (rotated image has swapped width/height)

---

## Phase 5: Optimization and Integration (2-3 hours)

### Task 5.1: Shared Memory Blur

Optimize Gaussian blur using shared memory:

```cuda
#define TILE_SIZE 16
#define FILTER_RADIUS 1

__global__ void gaussian_blur_shared(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    // Shared memory tile (includes halo region)
    __shared__ float tile[TILE_SIZE + 2*FILTER_RADIUS][TILE_SIZE + 2*FILTER_RADIUS][3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    // Load tile + halo region
    for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
        for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
            int load_x = x + dx;
            int load_y = y + dy;

            // Clamp to image boundaries
            load_x = min(max(load_x, 0), width - 1);
            load_y = min(max(load_y, 0), height - 1);

            int idx = (load_y * width + load_x) * channels;
            for (int c = 0; c < channels; c++) {
                tile[ty + FILTER_RADIUS + dy][tx + FILTER_RADIUS + dx][c] = input[idx + c];
            }
        }
    }
    __syncthreads();

    // Apply Gaussian kernel using shared memory
    if (x < width && y < height) {
        const float kernel[9] = {
            1.0f/16, 2.0f/16, 1.0f/16,
            2.0f/16, 4.0f/16, 2.0f/16,
            1.0f/16, 2.0f/16, 1.0f/16
        };

        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += kernel[(ky+1)*3 + (kx+1)] *
                           tile[ty + FILTER_RADIUS + ky][tx + FILTER_RADIUS + kx][c];
                }
            }

            output[(y * width + x) * channels + c] = (unsigned char)sum;
        }
    }
}
```

**Expected speedup**: 2-3x over naive version!

### Task 5.2: Pipeline API

Create easy-to-use API for chaining operations:

```cuda
// src/pipeline.cu

class ImagePipeline {
private:
    Image* current;
    bool on_device;

public:
    ImagePipeline(const char* filename) {
        Image* h_img = load_image(filename);
        current = image_to_device(h_img);
        free_image(h_img);
        on_device = true;
    }

    ImagePipeline& grayscale() {
        // ... launch kernel ...
        return *this;
    }

    ImagePipeline& blur() {
        // ... launch kernel ...
        return *this;
    }

    ImagePipeline& brightness(float alpha, float beta) {
        // ... launch kernel ...
        return *this;
    }

    void save(const char* filename) {
        Image* h_img = image_to_host(current);
        save_image(filename, h_img);
        free_image(h_img);
    }

    ~ImagePipeline() {
        if (on_device) free_image(current);
    }
};

// Usage:
ImagePipeline("input.jpg")
    .blur()
    .grayscale()
    .brightness(1.2f, 10.0f)
    .save("output.png");
```

---

## Phase 6: Testing and Benchmarking (2-3 hours)

### Task 6.1: Correctness Tests

Compare against reference implementations:

```cuda
// tests/test_kernels.cu

bool test_grayscale() {
    // Load test image
    Image* img = load_image("images/test/lena.png");

    // CPU reference
    Image* cpu_gray = grayscale_cpu(img);

    // GPU version
    Image* d_img = image_to_device(img);
    Image* d_gray = grayscale_gpu(d_img);
    Image* gpu_gray = image_to_host(d_gray);

    // Compare (allow small differences due to floating-point)
    float max_diff = 0.0f;
    for (int i = 0; i < img->width * img->height; i++) {
        float diff = fabsf((float)cpu_gray->data[i] - (float)gpu_gray->data[i]);
        max_diff = fmaxf(max_diff, diff);
    }

    printf("Grayscale max difference: %.2f\n", max_diff);

    // Cleanup
    free_image(img);
    free_image(cpu_gray);
    free_image(gpu_gray);
    free_image(d_img);
    free_image(d_gray);

    return max_diff < 2.0f;  // Allow ±2 due to rounding
}
```

### Task 6.2: Performance Benchmarks

```cuda
// tests/benchmark.cu

void benchmark_kernel(const char* name, void (*kernel_func)(Image*, Image*),
                     Image* input) {
    Image* d_input = image_to_device(input);
    Image* d_output = allocate_image(input->width, input->height, input->channels);

    // Warm-up
    kernel_func(d_input, d_output);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel_func(d_input, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;

    printf("%-20s: %.3f ms\n", name, avg_ms);

    free_image(d_input);
    free_image(d_output);
}
```

**Target performance** (1920×1080 image):
- Grayscale: < 0.5 ms
- Brightness: < 0.6 ms
- Blur (3×3): < 1.5 ms
- Sobel edge: < 1.0 ms
- Flip: < 0.4 ms

---

## Bonus Challenges

### Challenge 1: Batch Processing

Process multiple images in parallel:
```cuda
__global__ void batch_grayscale_kernel(
    const unsigned char** inputs,  // Array of image pointers
    unsigned char** outputs,
    int* widths,
    int* heights,
    int batch_size
) {
    int img_id = blockIdx.z;  // 3D grid!
    if (img_id >= batch_size) return;

    // ... process inputs[img_id] ...
}
```

### Challenge 2: Real-time Video

Integrate with OpenCV for webcam processing:
```cpp
#include <opencv2/opencv.hpp>

cv::VideoCapture cap(0);  // Webcam
while (true) {
    cv::Mat frame;
    cap >> frame;

    // Transfer to GPU
    Image* d_img = mat_to_device_image(frame);

    // Apply filters
    ImagePipeline pipeline(d_img);
    pipeline.blur().edge_detect();

    // Display
    cv::Mat result = device_image_to_mat(pipeline.get());
    cv::imshow("CUDA Filters", result);
}
```

### Challenge 3: Advanced Filters

Implement more sophisticated filters:
- Bilateral filter (edge-preserving blur)
- Histogram equalization (contrast enhancement)
- Morphological operations (erosion, dilation)
- Color space conversions (HSV, LAB)

---

## Deliverables

### Required

1. ✅ Working kernels for 8+ operations
2. ✅ Image I/O utilities
3. ✅ Test suite with correctness verification
4. ✅ Benchmark suite with performance metrics
5. ✅ Example usage programs
6. ✅ README with compilation and usage instructions

### Optional

7. 🎁 Python bindings with PyTorch integration
8. 🎁 Real-time video processing demo
9. 🎁 Batch processing support
10. 🎁 Additional advanced filters

---

## Evaluation Criteria

### Functionality (40 points)
- All 8 kernels work correctly (5 points each)

### Performance (30 points)
- Grayscale: > 400 GB/s (10 points)
- Blur: Uses shared memory (10 points)
- Overall: 20x+ speedup vs CPU (10 points)

### Code Quality (20 points)
- Clean, readable code (5 points)
- Proper error handling (5 points)
- Reusable API design (5 points)
- Documentation (5 points)

### Testing (10 points)
- Correctness tests pass (5 points)
- Benchmarks run successfully (5 points)

**Total**: 100 points

**Target**: 80+ points for completion

---

## Learning Resources

### Tutorials
- [2D Thread Indexing](../../../tutorials/beginner/thread-hierarchy.md)
- [Memory Coalescing](../../../tutorials/intermediate/memory-coalescing.md)
- [Element-wise Operations](../../../tutorials/beginner/element-wise-operations.md)

### External Resources
- **STB Image Library**: https://github.com/nothings/stb (single-header image I/O)
- **Image Processing Basics**: https://en.wikipedia.org/wiki/Digital_image_processing
- **Kernel Design Patterns**: Computer Vision textbooks (Szeliski, etc.)

---

## Expected Outcomes

After completing this project:

**Technical Skills**:
- ✅ Proficient with 2D CUDA kernels
- ✅ Understand spatial data access patterns
- ✅ Can optimize memory-bound operations
- ✅ Know when to use shared memory

**Practical Skills**:
- ✅ Built a real-world GPU library
- ✅ Can process images 20-50x faster
- ✅ Understand filter design and implementation
- ✅ Ready for computer vision projects

**Performance Insights**:
- Element-wise: ~500-700 GB/s (80-90% peak)
- Spatial filters (naive): ~100-200 GB/s
- Spatial filters (optimized): ~300-400 GB/s
- Geometric transforms: ~400-600 GB/s

---

## Next Steps

After mastering image processing:

1. ✅ **[Reductions Tutorial](../../../tutorials/beginner/reductions.md)** - Aggregation operations
2. ✅ **[Convolution Project](../../intermediate/optimized-convolution/)** - Deep learning building block
3. ✅ **[Computer Vision Applications]** - Object detection, segmentation
4. ✅ **[Real-time Systems]** - Video processing, AR/VR

---

**Good luck with your image processing journey! 🚀📸**

---

**Project Designed by**: Project Designer Agent #3
**Session**: session_20251119_051648
**Difficulty**: Beginner
**Estimated Completion**: 10-15 hours
**Skills Practiced**: 2D kernels, spatial data, memory optimization, real-world application
