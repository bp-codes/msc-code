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


using Matrix = std::vector<std::vector<double>>;


// Serial matrix multiply
Matrix dgemm_openmp(const double k,
                 const Matrix& A,
                 const Matrix& B,
                 const double l,
                 const Matrix& C,
                 size_t rows,
                 size_t cols)
{
    omp_set_num_threads(1);

    // Basic dimension checks (as before)
    if (A.size() != rows || (rows && A[0].size() != cols)) throw std::invalid_argument("Matrix A must be rows x cols.");
    if (B.size() != cols || (cols && B[0].size() != rows)) throw std::invalid_argument("Matrix B must be cols x rows.");
    if (C.size() != rows || (rows && C[0].size() != rows)) throw std::invalid_argument("Matrix C must be rows x rows.");

    Matrix X(rows, std::vector<double>(rows, 0.0));

    // X = l * C
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; ++i) 
    {
        for (size_t j = 0; j < rows; ++j) 
        {
            X[i][j] = l * C[i][j];
        }
    }

    // X += k * A * B
    // Parallelize by rows of X (no write races across threads)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) 
    {
        auto& Xi       = X[i];        // row of X (write)
        const auto& Ai = A[i];        // row of A (read)
        for (size_t p = 0; p < cols; ++p) 
        {
            const double ka_ip = k * Ai[p];
            const auto& Bp     = B[p]; // row of B (read)
            // Vectorize the innermost accumulation across j
            #pragma omp simd
            for (size_t j = 0; j < rows; ++j) 
            {
                Xi[j] += ka_ip * Bp[j];
            }
        }
    }

    return X;
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



// X = k A * B + l C
int main(int argc, char** argv) 
{
    omp_set_num_threads(6); 

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



    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    // Test starts
    do 
    {
        Matrix X = dgemm_openmp(k, A, B, l, C, rows, cols);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);
    // Test ends


    // Clean up
    auto t2 = std::chrono::steady_clock::now();


    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    double calculated_value = check_sum(X);


    double time_setup = std::chrono::duration<double>(t1 - t0).count();
    double time_calc = std::chrono::duration<double>(t2 - t1).count();
    double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    double time_total = std::chrono::duration<double>(t3 - t0).count();
    double time_per_iteration = time_calc / iters;


    std::string method {"OpenMP"};
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
