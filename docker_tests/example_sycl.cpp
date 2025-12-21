#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        sycl::queue q;
        std::cout << "Running on: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << " ["
                  << q.get_device().get_info<sycl::info::device::vendor>()
                  << "]\n";

        constexpr std::size_t N = 1024;
        std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);

        {
            sycl::buffer<float> A(a.data(), sycl::range<1>(N));
            sycl::buffer<float> B(b.data(), sycl::range<1>(N));
            sycl::buffer<float> C(c.data(), sycl::range<1>(N));

            q.submit([&](sycl::handler &h) {
                auto accA = A.get_access<sycl::access::mode::read>(h);
                auto accB = B.get_access<sycl::access::mode::read>(h);
                auto accC = C.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    accC[i] = accA[i] + accB[i];
                });
            });
        }

        bool ok = true;
        for (std::size_t i = 0; i < N; ++i) {
            if (c[i] != 3.0f) {
                ok = false;
                break;
            }
        }

        std::cout << "Result check: " << (ok ? "OK" : "FAIL") << "\n";
        return ok ? 0 : 1;

    } catch (const sycl::exception &e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        return 1;
    }
}
 
