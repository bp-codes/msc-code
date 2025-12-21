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
#include <sycl/sycl.hpp>


using Matrix = std::vector<std::vector<double>>;


// Sycl matrix multiply
void dgemm_sycl(sycl::queue& q,
                  const double* sycldev_A,   // USM device, size rows*cols
                  const double* sycldev_B,   // USM device, size cols*rows
                  const double* sycldev_C,   // USM device, size rows*rows
                  double k,
                  double l,
                  size_t rows,
                  size_t cols,
                  double* sycldev_X)    // USM device, size rows*rows (output)
{

    const size_t M = rows; // rows of A and X
    const size_t K = cols; // cols of A, rows of B
    const size_t N = rows; // cols of B and X

    // Choose a local size (<= device limits). 16x16 = 256 threads is safe on RTX 3050.
    const size_t WG0 = 16, WG1 = 16;

    auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };
    const size_t G0 = ceil_div(M, WG0) * WG0;  // pad global size
    const size_t G1 = ceil_div(N, WG1) * WG1;

    q.submit([&](sycl::handler& h) {
        sycl::nd_range<2> nd({G0, G1}, {WG0, WG1});

        h.parallel_for(nd, [=](sycl::nd_item<2> it) {
            const size_t i = it.get_global_id(0);
            const size_t j = it.get_global_id(1);

            // Guard against padded threads
            if (i >= M || j >= N) return;

            double sum = 0.0;
            // A: MxK (row-major), B: KxN (row-major), C/X: MxN (row-major)
            for (size_t p = 0; p < K; ++p)
                sum += sycldev_A[i * K + p] * sycldev_B[p * N + j];

            sycldev_X[i * N + j] = k * sum + l * sycldev_C[i * N + j];
        });
    }).wait();

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

    sycl::queue q{ sycl::default_selector_v };
    std::cerr << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Flatten matrices
    std::vector<double> A_flat = flatten_matrix(A);
    std::vector<double> B_flat = flatten_matrix(B);
    std::vector<double> C_flat = flatten_matrix(C);



    // Allocate device memory once
    double* sycldev_A = sycl::malloc_device<double>(rows * cols, q);
    double* sycldev_B = sycl::malloc_device<double>(rows * cols, q);
    double* sycldev_C = sycl::malloc_device<double>(rows * rows, q);
    double* sycldev_X = sycl::malloc_device<double>(rows * rows, q);

    // Copy once
    q.memcpy(sycldev_A, A_flat.data(), A_flat.size() * sizeof(double)).wait();
    q.memcpy(sycldev_B, B_flat.data(), B_flat.size() * sizeof(double)).wait();
    q.memcpy(sycldev_C, C_flat.data(), C_flat.size() * sizeof(double)).wait();

    
    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    // Test starts
    do 
    {
        dgemm_sycl(q, sycldev_A, sycldev_B, sycldev_C, k, l, rows, cols, sycldev_X);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);
    // Test ends


    // Clean up
    auto t2 = std::chrono::steady_clock::now();


    // Copy X from device and unflatten
    std::vector<double> X_flat(rows * cols);
    q.memcpy(X_flat.data(), sycldev_X, X_flat.size() * sizeof(double)).wait();
    X = unflatten_matrix(X_flat, rows, cols);


    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    double calculated_value = check_sum(X);


    double time_setup = std::chrono::duration<double>(t1 - t0).count();
    double time_calc = std::chrono::duration<double>(t2 - t1).count();
    double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    double time_total = std::chrono::duration<double>(t3 - t0).count();
    double time_per_iteration = time_calc / iters;


    std::string method {"SYCL"};
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
