// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

using Matrix = std::vector<std::vector<double>>;


Matrix multiply(const Matrix& A, const Matrix& B, size_t n) {
    Matrix C(n, std::vector<double>(n, 0)); // initialize result with zeros

    for (size_t i = 0; i < n; i++) 
    {
        for (size_t k = 0; k < n; k++) 
        {      
            for (size_t j = 0; j < n; j++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


void task(Matrix A, Matrix B)
{
    for(int i = 0; i <1; i++)
    {
        Matrix C = multiply(A, B, A.size());
        B = std::move(C);
    }
}



int main(int argc, char** argv) 
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <number>\n";
        return 1;
    }
    
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

    Matrix A(N, std::vector<double>(N, 0)); 
    Matrix B(N, std::vector<double>(N, 0)); 


    double last {};
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            double n = fmod(8039.0 * (last + i + j + 550607.0), 10000.0);
            A[i][j] = n;
            last = n;
        }
    }

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            double n = fmod(8039.0 * (last + i + j + 550607.0), 10000.0);
            B[i][j] = n;
            last = n;
        }
    }
    

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    do 
    {
        task(A, B);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);
   
    std::cout << iters << std::endl;
    
    return 0;
}
