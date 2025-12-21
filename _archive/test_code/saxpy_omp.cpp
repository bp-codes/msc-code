#include <cstdio>
#include <vector>
#include <cmath>
#include <omp.h>

int main() {
  const int N = 1 << 20;
  const float a = 2.0f;
  std::vector<float> x(N, 1.5f), y(N, 2.0f);

  // Make sure offloading is required (fail fast if GPU isn’t available)
  omp_set_default_device(0);

  // Map data to device and compute: y = a*x + y
  #pragma omp target teams distribute parallel for map(to:x[0:N]) map(tofrom:y[0:N])
  for (int i = 0; i < N; ++i) {
    y[i] = a * x[i] + y[i];
  }

  // Quick check
  float err = std::fabs(y[0] - (a*1.5f + 2.0f));
  std::printf("y[0]=%f (expected %f), err=%g\n", y[0], a*1.5f + 2.0f, err);
  return err < 1e-6 ? 0 : 1;
}
 
