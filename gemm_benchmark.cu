/***************************************************************************************************
 * gemm_benchmark.cu
 *
 * Production-quality GEMM benchmark for measuring HPC device performance in TFLOPS.
 *
 * Features:
 *   - Multiple GEMM variants: FP32 (CUDA cores), FP16 Tensor Cores, BF16 Tensor Cores
 *   - Proper warmup iterations to exclude compilation overhead
 *   - Multiple timed runs with statistical analysis (mean, min, max)
 *   - Correct data type conversions
 *   - Accurate TFLOPS calculations
 *   - Matrix size sweep from small to large
 *   - Device capability detection
 *   - Correctness validation against CPU reference
 *   - CSV output for analysis
 *   - Memory bandwidth reporting
 *
 * Compile:
 *   nvcc gemm_benchmark_complete.cu -O3 --gpu-architecture=sm_89 -o gemm_benchmark
 *   (Adjust sm_89 to match your GPU architecture: sm_70 for V100, sm_80 for A100, sm_89 for H100)
 *
 * Run:
 *   ./gemm_benchmark
 *
 * Author: Claude Code, liy
 * Date: 2026
 ***************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <map>

using namespace nvcuda;

// ================================================================================================
// Configuration Constants
// ================================================================================================

#define TILE_SIZE 16           // Shared memory tile size for FP32 GEMM
#define WMMA_M 16              // Tensor Core tile dimensions
#define WMMA_N 16
#define WMMA_K 16

#define WARMUP_ITERS 5         // Number of warmup iterations (exclude from timing)
#define BENCHMARK_ITERS 20     // Number of timed iterations for averaging

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ================================================================================================
// Kernel 1: FP32 Shared Memory GEMM (CUDA Cores)
// ================================================================================================

__global__ void gemm_fp32_kernel(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        // Known issue: Bank conflicts can occur here, but we keep the benchmark simple
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            Asub[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < K && col < N)
            Bsub[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ================================================================================================
// Kernel 2: FP16 Tensor Core GEMM
// ================================================================================================

__global__ void gemm_fp16_tensorcore_kernel(const half *A, const half *B, float *C,
                                            int M, int N, int K) {
    // Calculate warp ID within the grid
    // Each block contains blockDim.x * blockDim.y / 32 warps
    // Each warp (32 threads) computes one 16x16 output tile
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warpsPerBlock = (blockDim.x * blockDim.y) / 32;

    // Calculate which 16x16 tile this warp is responsible for
    int warpM = blockIdx.y * (blockDim.y / 32) + (warpId / (blockDim.x / 32));
    int warpN = blockIdx.x * (blockDim.x / 32) + (warpId % (blockDim.x / 32));

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N)
        return;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Main loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        const half *tileA = A + (warpM * WMMA_M) * K + i;
        const half *tileB = B + i * N + (warpN * WMMA_N);

        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    float *tileC = C + (warpM * WMMA_M) * N + (warpN * WMMA_N);
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

// ================================================================================================
// Kernel 3: BF16 Tensor Core GEMM (requires sm_80+)
// ================================================================================================

__global__ void gemm_bf16_tensorcore_kernel(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                                            float *C, int M, int N, int K) {
#if __CUDA_ARCH__ >= 800
    // Calculate warp ID within the grid (same pattern as FP16 kernel)
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warpsPerBlock = (blockDim.x * blockDim.y) / 32;

    int warpM = blockIdx.y * (blockDim.y / 32) + (warpId / (blockDim.x / 32));
    int warpN = blockIdx.x * (blockDim.x / 32) + (warpId % (blockDim.x / 32));

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N)
        return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < K; i += WMMA_K) {
        const __nv_bfloat16 *tileA = A + (warpM * WMMA_M) * K + i;
        const __nv_bfloat16 *tileB = B + i * N + (warpN * WMMA_N);

        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float *tileC = C + (warpM * WMMA_M) * N + (warpN * WMMA_N);
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
#endif
}

// ================================================================================================
// cuBLAS GEMM Wrappers
// ================================================================================================

// cuBLAS FP32 GEMM: C = A × B
void cublas_gemm_fp32(cublasHandle_t handle, const float *A, const float *B, float *C,
                      int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major, so we compute: C = B^T × A^T to get row-major result
    // C (M×N) = A (M×K) × B (K×N)
    // In column-major: C^T (N×M) = B^T (N×K) × A^T (K×M)
    CUBLAS_CHECK(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            B, N,
                            A, K,
                            &beta,
                            C, N));
}

// cuBLAS FP16 GEMM with FP32 accumulation: C = A × B
void cublas_gemm_fp16(cublasHandle_t handle, const half *A, const half *B, float *C,
                      int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cublasGemmEx for mixed precision
    CUBLAS_CHECK(cublasGemmEx(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, CUDA_R_16F, N,
                             A, CUDA_R_16F, K,
                             &beta,
                             C, CUDA_R_32F, N,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// cuBLAS BF16 GEMM with FP32 accumulation: C = A × B
void cublas_gemm_bf16(cublasHandle_t handle, const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                      float *C, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, CUDA_R_16BF, N,
                             A, CUDA_R_16BF, K,
                             &beta,
                             C, CUDA_R_32F, N,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ================================================================================================
// Helper Functions
// ================================================================================================

// CPU reference GEMM for validation
void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Convert float array to half
void float_to_half(const float *src, half *dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2half(src[i]);
    }
}

// Convert float array to bfloat16
void float_to_bfloat16(const float *src, __nv_bfloat16 *dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// Transpose matrix for column-major layout (needed for Tensor Core B matrix)
void transpose_matrix(const float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// Calculate max absolute error
float compute_max_error(const float *ref, const float *result, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = fabsf(ref[i] - result[i]);
        max_err = fmaxf(max_err, err);
    }
    return max_err;
}

// Statistical helpers
float compute_mean(const std::vector<float> &times) {
    return std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
}

float compute_min(const std::vector<float> &times) {
    return *std::min_element(times.begin(), times.end());
}

float compute_max(const std::vector<float> &times) {
    return *std::max_element(times.begin(), times.end());
}

// ================================================================================================
// Benchmark Runner
// ================================================================================================

struct BenchmarkResult {
    std::string variant_name;
    int matrix_size;
    float time_mean_ms;
    float time_min_ms;
    float time_max_ms;
    double tflops;
    float max_error;
    double memory_bandwidth_gb;
};

template<typename T>
void run_benchmark(const std::string &name,
                   int variant_id,
                   const T *d_A, const T *d_B, float *d_C,
                   int M, int N, int K,
                   dim3 grid, dim3 block,
                   const float *h_ref,
                   std::vector<BenchmarkResult> &results) {

    std::cout << "\n" << name << "..." << std::endl;

    // Warmup runs
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (variant_id == 0) {
            gemm_fp32_kernel<<<grid, block>>>(
                reinterpret_cast<const float*>(d_A),
                reinterpret_cast<const float*>(d_B),
                d_C, M, N, K);
        } else if (variant_id == 1) {
            gemm_fp16_tensorcore_kernel<<<grid, block>>>(
                reinterpret_cast<const half*>(d_A),
                reinterpret_cast<const half*>(d_B),
                d_C, M, N, K);
        } else if (variant_id == 2) {
            gemm_bf16_tensorcore_kernel<<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(d_A),
                reinterpret_cast<const __nv_bfloat16*>(d_B),
                d_C, M, N, K);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        if (variant_id == 0) {
            gemm_fp32_kernel<<<grid, block>>>(
                reinterpret_cast<const float*>(d_A),
                reinterpret_cast<const float*>(d_B),
                d_C, M, N, K);
        } else if (variant_id == 1) {
            gemm_fp16_tensorcore_kernel<<<grid, block>>>(
                reinterpret_cast<const half*>(d_A),
                reinterpret_cast<const half*>(d_B),
                d_C, M, N, K);
        } else if (variant_id == 2) {
            gemm_bf16_tensorcore_kernel<<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(d_A),
                reinterpret_cast<const __nv_bfloat16*>(d_B),
                d_C, M, N, K);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Statistics
    float time_mean = compute_mean(times);
    float time_min = compute_min(times);
    float time_max = compute_max(times);

    // Calculate TFLOPS (2*M*N*K operations)
    double total_flops = 2.0 * M * N * K;
    double tflops = total_flops / (time_mean * 1e-3) / 1e12;

    // Calculate memory bandwidth (GB/s)
    // Data read: M*K + K*N elements, data written: M*N elements
    size_t element_size = (variant_id == 0) ? sizeof(float) : sizeof(half);
    double bytes_accessed = (M * K + K * N) * element_size + M * N * sizeof(float);
    double memory_bandwidth_gb = bytes_accessed / (time_mean * 1e-3) / 1e9;

    // Validate correctness
    std::vector<float> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float max_error = compute_max_error(h_ref, h_C.data(), M * N);

    // Print results
    std::cout << "  Time (mean): " << std::fixed << std::setprecision(3)
              << time_mean << " ms  [min: " << time_min << ", max: " << time_max << "]" << std::endl;
    std::cout << "  Performance: " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
    std::cout << "  Memory BW:   " << std::setprecision(1) << memory_bandwidth_gb << " GB/s" << std::endl;
    std::cout << "  Max Error:   " << std::scientific << std::setprecision(2) << max_error << std::endl;

    // Store result
    BenchmarkResult result;
    result.variant_name = name;
    result.matrix_size = N;
    result.time_mean_ms = time_mean;
    result.time_min_ms = time_min;
    result.time_max_ms = time_max;
    result.tflops = tflops;
    result.max_error = max_error;
    result.memory_bandwidth_gb = memory_bandwidth_gb;
    results.push_back(result);
}

// cuBLAS benchmark runner (separate because it doesn't use kernel launch parameters)
template<typename T>
void run_cublas_benchmark(const std::string &name,
                          int variant_id,
                          cublasHandle_t handle,
                          const T *d_A, const T *d_B, float *d_C,
                          int M, int N, int K,
                          const float *h_ref,
                          std::vector<BenchmarkResult> &results) {

    std::cout << "\n" << name << "..." << std::endl;

    // Warmup runs
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (variant_id == 3) {  // FP32 cuBLAS
            cublas_gemm_fp32(handle,
                            reinterpret_cast<const float*>(d_A),
                            reinterpret_cast<const float*>(d_B),
                            d_C, M, N, K);
        } else if (variant_id == 4) {  // FP16 cuBLAS
            cublas_gemm_fp16(handle,
                            reinterpret_cast<const half*>(d_A),
                            reinterpret_cast<const half*>(d_B),
                            d_C, M, N, K);
        } else if (variant_id == 5) {  // BF16 cuBLAS
            cublas_gemm_bf16(handle,
                            reinterpret_cast<const __nv_bfloat16*>(d_A),
                            reinterpret_cast<const __nv_bfloat16*>(d_B),
                            d_C, M, N, K);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        if (variant_id == 3) {
            cublas_gemm_fp32(handle,
                            reinterpret_cast<const float*>(d_A),
                            reinterpret_cast<const float*>(d_B),
                            d_C, M, N, K);
        } else if (variant_id == 4) {
            cublas_gemm_fp16(handle,
                            reinterpret_cast<const half*>(d_A),
                            reinterpret_cast<const half*>(d_B),
                            d_C, M, N, K);
        } else if (variant_id == 5) {
            cublas_gemm_bf16(handle,
                            reinterpret_cast<const __nv_bfloat16*>(d_A),
                            reinterpret_cast<const __nv_bfloat16*>(d_B),
                            d_C, M, N, K);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Statistics
    float time_mean = compute_mean(times);
    float time_min = compute_min(times);
    float time_max = compute_max(times);

    // Calculate TFLOPS
    double total_flops = 2.0 * M * N * K;
    double tflops = total_flops / (time_mean * 1e-3) / 1e12;

    // Calculate memory bandwidth
    size_t element_size = (variant_id == 3) ? sizeof(float) : sizeof(half);
    double bytes_accessed = (M * K + K * N) * element_size + M * N * sizeof(float);
    double memory_bandwidth_gb = bytes_accessed / (time_mean * 1e-3) / 1e9;

    // Validate correctness
    std::vector<float> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float max_error = compute_max_error(h_ref, h_C.data(), M * N);

    // Print results
    std::cout << "  Time (mean): " << std::fixed << std::setprecision(3)
              << time_mean << " ms  [min: " << time_min << ", max: " << time_max << "]" << std::endl;
    std::cout << "  Performance: " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
    std::cout << "  Memory BW:   " << std::setprecision(1) << memory_bandwidth_gb << " GB/s" << std::endl;
    std::cout << "  Max Error:   " << std::scientific << std::setprecision(2) << max_error << std::endl;

    // Store result
    BenchmarkResult result;
    result.variant_name = name;
    result.matrix_size = N;
    result.time_mean_ms = time_mean;
    result.time_min_ms = time_min;
    result.time_max_ms = time_max;
    result.tflops = tflops;
    result.max_error = max_error;
    result.memory_bandwidth_gb = memory_bandwidth_gb;
    results.push_back(result);
}

// ================================================================================================
// Main Benchmark Program
// ================================================================================================

int main() {
    // Print device information
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

	int memoryClockRate = 1, memoryBusWidth = 1;
	CUDA_CHECK(cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrClockRate, device));
	CUDA_CHECK(cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, device));

    std::cout << "========================================" << std::endl;
    std::cout << "      GEMM Performance Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Peak Memory Bandwidth: "
              << (2.0 * memoryClockRate * (memoryBusWidth / 8) / 1.0e6)
              << " GB/s" << std::endl;
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0)
              << " GB" << std::endl;
    std::cout << "Warmup iterations: " << WARMUP_ITERS << std::endl;
    std::cout << "Benchmark iterations: " << BENCHMARK_ITERS << std::endl;
    std::cout << "========================================" << std::endl;

    // Matrix sizes to test (square matrices)
    std::vector<int> sizes = {1024, 2048, 4096, 8192};

    // Add larger sizes if memory permits
    size_t max_size = 8192;
    size_t required_memory = 3 * max_size * max_size * sizeof(float);
    if (prop.totalGlobalMem > required_memory * 2) {
        sizes.push_back(16384);
    }

    std::vector<BenchmarkResult> all_results;

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Enable Tensor Core math mode for cuBLAS (for FP16/BF16)
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    for (int size : sizes) {
        int M = size, N = size, K = size;

        std::cout << "\n========================================" << std::endl;
        std::cout << "Matrix Size: " << M << " x " << N << " x " << K << std::endl;
        std::cout << "========================================" << std::endl;

        // Host memory allocation
        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        std::vector<float> h_B_transposed(K * N);
        std::vector<float> h_C_ref(M * N);

        // Initialize with random values
        srand(42);  // Fixed seed for reproducibility
        for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

        // Transpose B for column-major layout (needed for Tensor Cores)
        transpose_matrix(h_B.data(), h_B_transposed.data(), K, N);

        // CPU reference (only for smaller matrices to save time)
        std::cout << "\nComputing CPU reference..." << std::flush;
        if (size <= 2048) {
            cpu_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
            std::cout << " Done." << std::endl;
        } else {
            std::cout << " Skipped (matrix too large)." << std::endl;
            // For large matrices, just use first GPU result as reference
            std::fill(h_C_ref.begin(), h_C_ref.end(), 0.0f);
        }

        // =========================================================================================
        // Variant 1: FP32 CUDA Cores
        // =========================================================================================
        {
            float *d_A32, *d_B32, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A32, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_B32, K * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_A32, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B32, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

            run_benchmark("FP32 CUDA Cores", 0, d_A32, d_B32, d_C, M, N, K,
                         grid, block, h_C_ref.data(), all_results);

            // If we skipped CPU reference, use this as reference
            if (size > 2048) {
                CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
            }

            CUDA_CHECK(cudaFree(d_A32));
            CUDA_CHECK(cudaFree(d_B32));
            CUDA_CHECK(cudaFree(d_C));
        }

        // =========================================================================================
        // Variant 2: FP16 Tensor Cores
        // =========================================================================================
        if (prop.major >= 7) {  // Tensor Cores available on sm_70+
            half *d_A16, *d_B16;
            float *d_C;
            CUDA_CHECK(cudaMalloc(&d_A16, M * K * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_B16, K * N * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

            // Convert to FP16 on host, then copy
            std::vector<half> h_A16(M * K);
            std::vector<half> h_B16(K * N);
            float_to_half(h_A.data(), h_A16.data(), M * K);
            float_to_half(h_B_transposed.data(), h_B16.data(), K * N);

            CUDA_CHECK(cudaMemcpy(d_A16, h_A16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B16, h_B16.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

            // Each block has 128 threads = 4 warps
            // Each warp computes one 16x16 tile
            // Block computes 2x2 tiles (32x32 output)
            dim3 block(64, 2);  // 128 threads = 4 warps
            dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32);

            run_benchmark("FP16 Tensor Cores", 1, d_A16, d_B16, d_C, M, N, K,
                         grid, block, h_C_ref.data(), all_results);

            CUDA_CHECK(cudaFree(d_A16));
            CUDA_CHECK(cudaFree(d_B16));
            CUDA_CHECK(cudaFree(d_C));
        }

        // =========================================================================================
        // Variant 3: BF16 Tensor Cores (Ampere and newer)
        // =========================================================================================
        if (prop.major >= 8) {  // BF16 Tensor Cores available on sm_80+
            __nv_bfloat16 *d_Abf16, *d_Bbf16;
            float *d_C;
            CUDA_CHECK(cudaMalloc(&d_Abf16, M * K * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_Bbf16, K * N * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

            // Convert to BF16 on host, then copy
            std::vector<__nv_bfloat16> h_Abf16(M * K);
            std::vector<__nv_bfloat16> h_Bbf16(K * N);
            float_to_bfloat16(h_A.data(), h_Abf16.data(), M * K);
            float_to_bfloat16(h_B_transposed.data(), h_Bbf16.data(), K * N);

            CUDA_CHECK(cudaMemcpy(d_Abf16, h_Abf16.data(), M * K * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Bbf16, h_Bbf16.data(), K * N * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice));

            // Same configuration as FP16: 4 warps per block
            dim3 block(64, 2);  // 128 threads = 4 warps
            dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32);

            run_benchmark("BF16 Tensor Cores", 2, d_Abf16, d_Bbf16, d_C, M, N, K,
                         grid, block, h_C_ref.data(), all_results);

            CUDA_CHECK(cudaFree(d_Abf16));
            CUDA_CHECK(cudaFree(d_Bbf16));
            CUDA_CHECK(cudaFree(d_C));
        }

        // =========================================================================================
        // cuBLAS Variants (Gold Standard Baseline)
        // =========================================================================================
        std::cout << "\n========================================" << std::endl;
        std::cout << "      cuBLAS (NVIDIA Optimized)" << std::endl;
        std::cout << "========================================" << std::endl;

        // =========================================================================================
        // cuBLAS FP32
        // =========================================================================================
        {
            float *d_A32, *d_B32, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A32, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_B32, K * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_A32, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B32, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

            run_cublas_benchmark("cuBLAS FP32", 3, cublas_handle, d_A32, d_B32, d_C, M, N, K,
                                h_C_ref.data(), all_results);

            CUDA_CHECK(cudaFree(d_A32));
            CUDA_CHECK(cudaFree(d_B32));
            CUDA_CHECK(cudaFree(d_C));
        }

        // =========================================================================================
        // cuBLAS FP16 Tensor Cores
        // =========================================================================================
        if (prop.major >= 7) {
            half *d_A16, *d_B16;
            float *d_C;
            CUDA_CHECK(cudaMalloc(&d_A16, M * K * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_B16, K * N * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

            // Convert to FP16
            std::vector<half> h_A16(M * K);
            std::vector<half> h_B16(K * N);
            float_to_half(h_A.data(), h_A16.data(), M * K);
            float_to_half(h_B.data(), h_B16.data(), K * N);

            CUDA_CHECK(cudaMemcpy(d_A16, h_A16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B16, h_B16.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

            run_cublas_benchmark("cuBLAS FP16 Tensor Cores", 4, cublas_handle, d_A16, d_B16, d_C, M, N, K,
                                h_C_ref.data(), all_results);

            CUDA_CHECK(cudaFree(d_A16));
            CUDA_CHECK(cudaFree(d_B16));
            CUDA_CHECK(cudaFree(d_C));
        }

        // =========================================================================================
        // cuBLAS BF16 Tensor Cores
        // =========================================================================================
        if (prop.major >= 8) {
            __nv_bfloat16 *d_Abf16, *d_Bbf16;
            float *d_C;
            CUDA_CHECK(cudaMalloc(&d_Abf16, M * K * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_Bbf16, K * N * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

            // Convert to BF16
            std::vector<__nv_bfloat16> h_Abf16(M * K);
            std::vector<__nv_bfloat16> h_Bbf16(K * N);
            float_to_bfloat16(h_A.data(), h_Abf16.data(), M * K);
            float_to_bfloat16(h_B.data(), h_Bbf16.data(), K * N);

            CUDA_CHECK(cudaMemcpy(d_Abf16, h_Abf16.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Bbf16, h_Bbf16.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

            run_cublas_benchmark("cuBLAS BF16 Tensor Cores", 5, cublas_handle, d_Abf16, d_Bbf16, d_C, M, N, K,
                                h_C_ref.data(), all_results);

            CUDA_CHECK(cudaFree(d_Abf16));
            CUDA_CHECK(cudaFree(d_Bbf16));
            CUDA_CHECK(cudaFree(d_C));
        }
    }

    // Destroy cuBLAS handle
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    // =========================================================================================
    // Save results to CSV
    // =========================================================================================
    std::ofstream csv("gemm_benchmark_results.csv");
    csv << "Variant,MatrixSize,TimeMean_ms,TimeMin_ms,TimeMax_ms,TFLOPS,MemoryBW_GB_s,MaxError\n";

    for (const auto &r : all_results) {
        csv << r.variant_name << ","
            << r.matrix_size << ","
            << std::fixed << std::setprecision(3) << r.time_mean_ms << ","
            << r.time_min_ms << ","
            << r.time_max_ms << ","
            << std::setprecision(2) << r.tflops << ","
            << std::setprecision(1) << r.memory_bandwidth_gb << ","
            << std::scientific << std::setprecision(2) << r.max_error << "\n";
    }
    csv.close();

    // =========================================================================================
    // Summary
    // =========================================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "           BENCHMARK SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;

    // Find peak performance for each variant
    std::map<std::string, double> peak_tflops;
    for (const auto &r : all_results) {
        if (peak_tflops.find(r.variant_name) == peak_tflops.end() ||
            r.tflops > peak_tflops[r.variant_name]) {
            peak_tflops[r.variant_name] = r.tflops;
        }
    }

    std::cout << "\nPeak Performance:" << std::endl;
    for (const auto &p : peak_tflops) {
        std::cout << "  " << std::left << std::setw(25) << p.first << ": "
                  << std::fixed << std::setprecision(2) << p.second << " TFLOPS" << std::endl;
    }

    std::cout << "\nResults saved to: gemm_benchmark_results.csv" << std::endl;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
