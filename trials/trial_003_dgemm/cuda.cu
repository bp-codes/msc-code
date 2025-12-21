// openmp.cpp

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <cuda_runtime.h>



#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      std::abort();                                                            \
    }                                                                          \
  } while (0)



using Matrix = std::vector<std::vector<double>>;


__global__ void dgemm_kernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             const double* __restrict__ C,
                             double* __restrict__ X,
                             double k, double l,
                             int M, int K, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row in X
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col in X
    if (i >= M || j >= N) return;

    double sum = 0.0;
    // row-major indexing:
    // A[i, p] => A[i*K + p]
    // B[p, j] => B[p*N + j]
    for (int p = 0; p < K; ++p) {
        sum += A[i * K + p] * B[p * N + j];
    }
    // C[i, j] => C[i*N + j]
    X[i * N + j] = k * sum + l * C[i * N + j];
}

inline void dgemm_cuda(const double* d_A,
                       const double* d_B,
                       const double* d_C,
                       double* d_X,
                       double k, double l,
                       int rows, int cols,
                       cudaStream_t stream = 0)
{
    const int M = rows;
    const int K = cols;
    const int N = rows;

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    dgemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, d_X, k, l, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}




// Serial matrix multiply
Matrix dgemm_serial(const double k,
                    const Matrix& A,
                    const Matrix& B,
                    const double l,
                    const Matrix& C,
                    size_t rows,
                    size_t cols)
{
    // Basic dimension checks
    if (A.size() != rows || (rows && A[0].size() != cols)) throw std::invalid_argument("Matrix A must be rows x cols.");
    if (B.size() != cols || (cols && B[0].size() != rows)) throw std::invalid_argument("Matrix B must be cols x rows.");
    if (C.size() != rows || (rows && C[0].size() != rows)) throw std::invalid_argument("Matrix C must be rows x rows.");

    // Initialize X = l * C
    Matrix X(rows, std::vector<double>(rows, 0.0));
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < rows; ++j)
        {
            X[i][j] = l * C[i][j];
        }
    }

    // Compute k * A * B and add to X
    for (size_t i = 0; i < rows; ++i) 
    {
        for (size_t p = 0; p < cols; ++p) 
        {            
            const double ka_ip =  k * A[i][p];
            for (size_t j = 0; j < rows; ++j) 
            {
                X[i][j] += ka_ip * B[p][j];
            }
        }
    }

    return X;
}



// Serial - sum matrix C
double check_sum(const Matrix& X)
{
    double sum {};

    for (size_t i = 0; i < X.size(); i++) 
    {
        for (size_t j = 0; j < X[i].size(); j++) 
        {      
            sum += X[i][j];
        }
    }
    return sum;
}


std::vector<double> flatten_matrix(const Matrix& matrix)
{
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<double> flat(rows * cols);
    for (size_t i = 0; i < rows; i++)
    {
        std::copy(matrix[i].begin(), matrix[i].end(), flat.begin() + i * cols);
    }
    return flat;
}


static inline Matrix unflatten_matrix(const std::vector<double>& flat,
                                      size_t rows, size_t cols)
{
    Matrix M(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i)
        std::copy(flat.begin() + i * cols, flat.begin() + (i + 1) * cols, M[i].begin());
    return M;
}



// X = k A * B + l C
int main(int argc, char** argv) 
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " test_time rows cols\n";
        return 1;
    }
    
    double test_time = std::atof(argv[1]);
    const int rows = std::atoi(argv[2]);   // was N
    const int cols = std::atoi(argv[3]);   // was M

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Matrix dimensions:
    // A: rows × cols
    // B: cols × rows 
    // C: rows × rows  
    Matrix A(rows, std::vector<double>(cols, 0.0)); 
    Matrix B(cols, std::vector<double>(rows, 0.0)); 
    Matrix C(rows, std::vector<double>(rows, 0.0)); 

    double k = dist(rng);
    double l = dist(rng);

    // Fill A
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            A[i][j] = dist(rng);
        }
    }

    // Fill B
    for (int i = 0; i < cols; i++) 
    {        
        for (int j = 0; j < rows; j++) 
        {
            B[i][j] = dist(rng);
        }
    }

    // Fill C
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < rows; j++) 
        {
            C[i][j] = dist(rng);
        }
    }


    Matrix X = dgemm_serial(k, A, B, l, C, rows, cols);
    double expected_value = check_sum(X);

    
    // ======= Calculation Starts ========

    // Setup
    auto t0 = std::chrono::steady_clock::now();

    // (Optional) pick device 0
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "Using device: " << prop.name << "\n";

    // Flatten matrices
    std::vector<double> A_flat = flatten_matrix(A);
    std::vector<double> B_flat = flatten_matrix(B);
    std::vector<double> C_flat = flatten_matrix(C);

    // Allocate device memory once
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    double* d_X = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, A_flat.size() * sizeof(double)));          // rows*cols
    CUDA_CHECK(cudaMalloc(&d_B, B_flat.size() * sizeof(double)));          // cols*rows
    CUDA_CHECK(cudaMalloc(&d_C, C_flat.size() * sizeof(double)));          // rows*rows
    CUDA_CHECK(cudaMalloc(&d_X, static_cast<size_t>(rows) * rows * sizeof(double)));

    // Copy once (async copies are fine; sync before timing loop if needed)
    CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(),
                        A_flat.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_flat.data(),
                        B_flat.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C_flat.data(),
                        C_flat.size() * sizeof(double),
                        cudaMemcpyHostToDevice));

    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    // Test starts
    do {
        dgemm_cuda(d_A, d_B, d_C, d_X, k, l, rows, cols, /*stream=*/0);
        // Ensure the iteration completed before counting it
        CUDA_CHECK(cudaDeviceSynchronize());
        ++iters;
    } while (std::chrono::steady_clock::now() < deadline);
    // Test ends

    auto t2 = std::chrono::steady_clock::now();

    // Copy X from device and unflatten
    std::vector<double> X_flat(static_cast<size_t>(rows) * rows);  // NOTE: rows*rows, not rows*cols
    CUDA_CHECK(cudaMemcpy(X_flat.data(), d_X,
                        X_flat.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
    X = unflatten_matrix(X_flat, rows, rows);

    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_X));


    double calculated_value = check_sum(X);


    double time_setup = std::chrono::duration<double>(t1 - t0).count();
    double time_calc = std::chrono::duration<double>(t2 - t1).count();
    double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    double time_total = std::chrono::duration<double>(t3 - t0).count();
    double time_per_iteration = time_calc / iters;


    std::string method {"CUDA"};
    std::string comments {"operation:DGEMM"};
    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-9;

    std::cout << method << "," 
              << expected_value << "," 
              << calculated_value << "," 
              << iters << "," 
              << time_per_iteration << "," 
              << time_setup << "," 
              << time_calc << "," 
              << time_cleanup << "," 
              << time_total << "," 
              << passed_check << "," 
              << comments << "" 
              << std::endl;

    
    return 0;
}
