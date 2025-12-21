#include <iostream>
#include <cmath>
#include <chrono>
#include <CL/sycl.hpp>

int main() {

    auto start_setup = std::chrono::high_resolution_clock::now();
    //############################################################

    constexpr int N = 1024;

    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    // Deterministic initialization
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (i + j) % 100;
            B[i * N + j] = (i * j + 3) % 100;
        }
    }

    auto end_setup = std::chrono::high_resolution_clock::now();



    auto start_calc = std::chrono::high_resolution_clock::now();
    //############################################################

    sycl::queue q{sycl::default_selector{}};

    sycl::buffer<double, 1> bufA(A.data(), sycl::range<1>(N * N));
    sycl::buffer<double, 1> bufB(B.data(), sycl::range<1>(N * N));
    sycl::buffer<double, 1> bufC(C.data(), sycl::range<1>(N * N));

    q.submit([&](sycl::handler& h) {
        auto a = bufA.get_access<sycl::access::mode::read>(h);
        auto b = bufB.get_access<sycl::access::mode::read>(h);
        auto c = bufC.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>{N, N}, [=](sycl::id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += a[row * N + k] * b[k * N + col];
            c[row * N + col] = sum;
        });
    });

    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();
    
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
    {
        sum += C[i];
    }

    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << sum << std::endl;

    return 0;
}

