// sycl_mm.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <sycl/sycl.hpp>

using Matrix = std::vector<std::vector<double>>;

// ---- helpers to flatten/unflatten ----
static std::vector<double> flatten(const Matrix& M) {
    const size_t n = M.size();
    std::vector<double> out(n * n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            out[i * n + j] = M[i][j];
    return out;
}

static Matrix unflatten(const std::vector<double>& v, size_t n) {
    Matrix M(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            M[i][j] = v[i * n + j];
    return M;
}

// ---- SYCL matrix multiply: C = A * B ----
Matrix multiply_sycl(const Matrix& A_host, const Matrix& B_host, size_t n, sycl::queue& q) {
    // Flatten inputs
    std::vector<double> A = flatten(A_host);
    std::vector<double> B = flatten(B_host);
    std::vector<double> C(n * n, 0.0);

    {
        // Buffers wrap host memory; lifetime scope ensures synchronization on destruction
        sycl::buffer<double, 2> a_buf(A.data(), sycl::range<2>(n, n));
        sycl::buffer<double, 2> b_buf(B.data(), sycl::range<2>(n, n));
        sycl::buffer<double, 2> c_buf(sycl::range<2>(n, n));

        q.submit([&](sycl::handler& h) {
            sycl::accessor a(a_buf, h, sycl::read_only);
            sycl::accessor b(b_buf, h, sycl::read_only);
            sycl::accessor c(c_buf, h, sycl::write_only, sycl::no_init);

            h.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> idx) {
                const size_t i = idx[0];
                const size_t j = idx[1];
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            });
        });

        // Read back into C (host) via a host accessor
        {
            sycl::host_accessor c_host(c_buf, sycl::read_only);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    C[i * n + j] = c_host[i][j];
        }
        // buffers go out of scope here -> implicit wait for completion
    }

    return unflatten(C, n);
}

void task(Matrix A, Matrix B, sycl::queue& q)
{
    for (int i = 0; i < 1; i++) {
        Matrix C = multiply_sycl(A, B, A.size(), q);
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

    // Prefer GPU; fall back to CPU if no GPU or no fp64 support
    sycl::queue q;
    try {
        q = sycl::queue{sycl::gpu_selector_v};
        // probe fp64 support
        if (!q.get_device().has(sycl::aspect::fp64)) {
            std::cerr << "GPU lacks fp64; falling back to CPU device.\n";
            q = sycl::queue{sycl::cpu_selector_v};
        }
    } catch (...) {
        q = sycl::queue{sycl::cpu_selector_v};
    }
    std::cerr << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";


    Matrix A(N, std::vector<double>(N, 0));
    Matrix B(N, std::vector<double>(N, 0));

    // Keep this initialization serial to preserve your 'last' dependency
    double last {};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double n = fmod(8039.0 * (last + i + j + 550607.0), 10000.0);
            A[i][j] = n;
            last = n;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double n = fmod(8039.0 * (last + i + j + 550607.0), 10000.0);
            B[i][j] = n;
            last = n;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    do {
        task(A, B, q);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);

    std::cout << iters << std::endl;
    return 0;
}
