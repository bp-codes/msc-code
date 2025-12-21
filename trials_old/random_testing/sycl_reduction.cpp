 
// sum_usm.cpp
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() 
{
    sycl::queue q{sycl::default_selector_v};
    std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

    constexpr size_t N = 1 << 20;
    std::vector<double> h(N, 1.0);

    // USM shared so host can read result without an explicit copy
    double* data = sycl::malloc_shared<double>(N, q);
    double* sum  = sycl::malloc_shared<double>(1, q);

    for (size_t i = 0; i < N; ++i) 
    {
        data[i] = h[i];
    }
    *sum = 0.0;

    q.parallel_for(
        sycl::range<1>(N),
        // initialize_to_identity => ignores the *previous* value of *sum
        sycl::reduction(sum, sycl::plus<>(),
                        sycl::property::reduction::initialize_to_identity{}),
        [=](sycl::id<1> i, auto& r) {
        r.combine(data[i]); // or: r += data[i]; (for plus<>)
        }).wait();

    std::cout << "sum = " << *sum << "\n";

    sycl::free(sum, q);
    sycl::free(data, q);
}
