// sycl_stress.cpp
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

static size_t getenv_size_t(const char* k, size_t def) {
  if (const char* v = std::getenv(k)) {
    try {
      return static_cast<size_t>(std::stoull(v));
    } catch (...) { return def; }
  }
  return def;
}
static double getenv_double(const char* k, double def) {
  if (const char* v = std::getenv(k)) {
    try {
      return std::stod(v);
    } catch (...) { return def; }
  }
  return def;
}

int main() {


    try {
      auto platforms = sycl::platform::get_platforms();

      for (const auto &platform : platforms) {
          std::cout << "Platform: " 
                    << platform.get_info<sycl::info::platform::name>() 
                    << "\n";

          auto devices = platform.get_devices();
          for (const auto &device : devices) {
              std::cout << "  Device: " 
                        << device.get_info<sycl::info::device::name>() 
                        << "\n";
              std::cout << "    Vendor: " 
                        << device.get_info<sycl::info::device::vendor>() 
                        << "\n";
              std::cout << "    Type: ";
              switch (device.get_info<sycl::info::device::device_type>()) {
                  case sycl::info::device_type::cpu: 
                      std::cout << "CPU"; break;
                  case sycl::info::device_type::gpu: 
                      std::cout << "GPU"; break;
                  case sycl::info::device_type::accelerator: 
                      std::cout << "Accelerator"; break;
                  default: 
                      std::cout << "Other"; break;
              }
              std::cout << "\n";
          }
          std::cout << std::endl;
      }
  } catch (sycl::exception const &e) {
      std::cerr << "SYCL Exception: " << e.what() << std::endl;
      return 1;
  }


  // Tunables (sane defaults for wide range of GPUs/CPUs)
  const size_t N      = getenv_size_t("SYCL_TEST_N",      1ull << 22); // 4,194,304
  const size_t INNER  = getenv_size_t("SYCL_TEST_INNER",  64);         // math ops per item
  const double TARGET = getenv_double("SYCL_TEST_TARGET_SECONDS", 6.0); // aim ~5–10s

  sycl::queue q;
  auto dev = q.get_device();
  std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
  std::cout << "N=" << N << ", INNER=" << INNER << ", target=" << TARGET << "s\n";

  // Use USM shared for simplicity/portability
  float* a = sycl::malloc_shared<float>(N, q);
  float* b = sycl::malloc_shared<float>(N, q);
  float* c = sycl::malloc_shared<float>(N, q);
  if (!a || !b || !c) {
    std::cerr << "USM allocation failed; reduce N via SYCL_TEST_N.\n";
    return 1;
  }

  // Initialize inputs
  q.parallel_for(N, [=](sycl::id<1> i){ a[i] = 1.0f; b[i] = 2.0f; }).wait();

  // For reductions we keep a scalar sum to avoid dead-code elimination
  double* sum_dev = sycl::malloc_shared<double>(1, q);
  *sum_dev = 0.0;

  // Round up global size to multiple of local size for performance
  const size_t local  = 256;
  const size_t groups = (N + local - 1) / local;
  const size_t global = groups * local;

  // Timer loop: keep launching kernels until we cross TARGET seconds
  size_t rounds = 0;
  auto t0 = std::chrono::steady_clock::now();
  while (true) {
    // Reset per-round reduction seed (not strictly needed, but clearer)
    double round_sum = 0.0;

    sycl::buffer<double,1> sum_buf(&round_sum, sycl::range<1>(1));
    q.submit([&](sycl::handler& h) {
      auto red = sycl::reduction(sum_buf, h, std::plus<double>());
      h.parallel_for(
        sycl::nd_range<1>(global, local), red,
        [=](sycl::nd_item<1> it, auto& sum) {
          const size_t i = it.get_global_id(0);
          // Guard extra items
          if (i >= N) return;

          // Load & compute a little "transcendental soup" to keep ALUs busy
          float x = a[i], y = b[i];
          // Prevent the compiler from trivially folding constants
          float t = static_cast<float>((i % 1024) + 1) * 1e-4f;

          // INNER controls per-item work; trig + fma-like math
          for (size_t k = 0; k < INNER; ++k) {
            x = sycl::sin(x + t) + sycl::cos(y - t) + x * 1.000001f;
            y = sycl::tan(x * 0.5f + t) + y * 0.999999f;
          }

          c[i] = x + y;
          sum.combine(static_cast<double>(c[i]));
        });
    }).wait();

    *sum_dev += round_sum; // host accumulation to keep a side-effect alive
    rounds++;

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - t0).count();
    if (elapsed >= TARGET) {
      std::cout << "Elapsed " << elapsed << " s after " << rounds << " kernel round(s)\n";
      break;
    }
  }

  // Touch results so the compiler can't elide anything
  double guard = *sum_dev;
  std::cout << "Checksum (ignore value): " << std::fixed << guard << "\n";

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  sycl::free(sum_dev, q);
  return 0;
}
 
