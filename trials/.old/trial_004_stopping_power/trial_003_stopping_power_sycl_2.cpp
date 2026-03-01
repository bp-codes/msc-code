// sycl_version.cpp
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <limits>

double stopping_power_Fe(double v_m_per_s) 
{
    // --- Physical constants (in compatible units) ---
    const double c = 299792458.0;                 // m/s
    const double me_MeV = 0.51099895000;          // electron mass energy, MeV
    const double K = 0.307075;                    // MeV·cm^2/g (Bethe constant)

    // --- Iron material constants ---
    const double Z = 26.0;                        // atomic number
    const double A = 55.845;                      // g/mol
    const double rho = 7.874;                     // g/cm^3
    const double I_eV = 286.0;                    // mean excitation energy, eV
    const double I_MeV = I_eV * 1.0e-6;           // convert eV -> MeV

    // --- Projectile (proton by default) ---
    const double z = 1.0;                         // charge state in |e|
    const double M_MeV = 938.2720813;             // proton rest energy, MeV

    // Guard: valid speed
    if (!(v_m_per_s > 0.0) || !(v_m_per_s < c)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Relativistic kinematics
    const double beta  = v_m_per_s / c;
    const double beta2 = beta * beta;
    const double gamma = 1.0 / sycl::sqrt(1.0 - beta2);
    const double me_over_M = me_MeV / M_MeV;

    // Tmax for heavy particle-electron kinematics (MeV)
    const double numerator = 2.0 * me_MeV * beta2 * gamma * gamma;
    const double denom = 1.0 + 2.0 * gamma * me_over_M + (me_over_M * me_over_M);
    const double Tmax = numerator / denom;

    // Argument inside the logarithm (dimensionless)
    const double logArg = (2.0 * me_MeV * beta2 * gamma * gamma * Tmax) / (I_MeV * I_MeV);
    if (!(logArg > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Mass stopping power (MeV·cm^2/g)
    const double prefactor = K * (Z / A) * (z * z) / beta2;
    const double bracket = 0.5 * sycl::log(logArg) - beta2; // (no density/shell corrections)
    const double S_over_rho = prefactor * bracket;

    // Linear stopping power in MeV/cm
    const double S_linear = S_over_rho * rho;
    return S_linear;
}

// Keeps the same signature; uses SYCL buffers + parallel_for internally.
void task(sycl::queue q, std::vector<double>& velocity_array, std::vector<double>& results,
          const size_t n, sycl::buffer<double,1>& bufVel, sycl::buffer<double,1>& bufRes)
{


    q.submit([&](sycl::handler& h) {
        auto vel = bufVel.get_access<sycl::access::mode::read>(h);
        auto res = bufRes.get_access<sycl::access::mode::write>(h);
        h.parallel_for<class StoppingPowerKernel>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            res[i] = stopping_power_Fe(vel[i]);
        });
    });
    q.wait(); // ensure completion before returning
}

int main(int argc, char** argv) 
{   
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

    std::vector<double> velocity_array;
    velocity_array.reserve(N);

    double last {};
    for (int i = 0; i < N; ++i) 
    {
        double n = std::fmod(1.23456789 * 8039.0 * (last + i + 550607.0), 10000.0);
        velocity_array.push_back(n);
        last = n;
    }

    std::vector<double> results(velocity_array.size());    

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    // Block scope 
    {
        static sycl::queue q{ sycl::default_selector_v };    
        const size_t n = velocity_array.size();
        if (results.size() != n) results.resize(n);

        sycl::buffer<double,1> bufVel(velocity_array.data(), sycl::range<1>(n));
        sycl::buffer<double,1> bufRes(results.data(),        sycl::range<1>(n));

        do 
        {
            task(q, velocity_array, results, n, bufVel, bufRes);
            iters++;
        } 
        while (std::chrono::steady_clock::now() < deadline);
    }
   
    std::cout << iters << "," << results[0] << "," << results[N-1] << std::endl;
    return 0;
}
