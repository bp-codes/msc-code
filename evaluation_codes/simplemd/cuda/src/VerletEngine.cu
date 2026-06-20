
#include "SimpleMD/Atom.hpp"
#include "SimpleMD/AtomPair.hpp"
#include "Maths/Vec3.hpp"
#include "SimpleMD/ConfigurationEngine.hpp"
#include "SimpleMD/VerletEngine.hpp"


// Kernels

#include "SimpleMD/VerletEngine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "SimpleMD/Atom.hpp"
#include "SimpleMD/AtomPair.hpp"
#include "SimpleMD/ConfigurationEngine.hpp"
#include "SimpleMD/Morse.hpp"
#include "SimpleMD/Timer.hpp"

namespace SimpleMD {

void cuda_check(cudaError_t err, const std::string& message)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(err));
    }
}

__global__ void zero_forces_kernel(Atom* atoms, std::size_t n_atoms)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_atoms) {
        return;
    }

    atoms[i].force.x = 0.0;
    atoms[i].force.y = 0.0;
    atoms[i].force.z = 0.0;
}

__global__ void calculate_force_kernel(Atom* atoms,
                                       const AtomPair* neighbour_list,
                                       std::size_t n_pairs,
                                       double r_cutoff)
{
    const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= n_pairs) {
        return;
    }

    const AtomPair pair = neighbour_list[k];

    if (pair.r > r_cutoff) {
        return;
    }

    const double force_mag = SimpleMD::Morse::force(pair.r, 0.343, 1.44, 2.863);

    const double fx = force_mag * pair.u_vec.x;
    const double fy = force_mag * pair.u_vec.y;
    const double fz = force_mag * pair.u_vec.z;

    const std::size_t i = pair.atom_i_idx;
    const std::size_t j = pair.atom_j_idx;

    atomicAdd(&atoms[i].force.x, -fx);
    atomicAdd(&atoms[i].force.y, -fy);
    atomicAdd(&atoms[i].force.z, -fz);

    atomicAdd(&atoms[j].force.x, fx);
    atomicAdd(&atoms[j].force.y, fy);
    atomicAdd(&atoms[j].force.z, fz);
}

__global__ void calculate_position_kernel(Atom* atoms,
                                          std::size_t n_atoms,
                                          double dt)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_atoms) {
        return;
    }

    Atom& atom = atoms[i];

    const double c = 0.5 * atom.inv_mass * dt * dt;

    atom.position.x += atom.velocity.x * dt + atom.force.x * c;
    atom.position.y += atom.velocity.y * dt + atom.force.y * c;
    atom.position.z += atom.velocity.z * dt + atom.force.z * c;

    // Fractional-position periodic boundary condition into [0, 1).
    atom.position.x -= floor(atom.position.x);
    atom.position.y -= floor(atom.position.y);
    atom.position.z -= floor(atom.position.z);
}

__global__ void store_force_kernel(Atom* atoms, std::size_t n_atoms)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_atoms) {
        return;
    }

    atoms[i].scratch.x = atoms[i].force.x;
    atoms[i].scratch.y = atoms[i].force.y;
    atoms[i].scratch.z = atoms[i].force.z;
}

__global__ void calculate_velocity_kernel(Atom* atoms,
                                          std::size_t n_atoms,
                                          double dt)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_atoms) {
        return;
    }

    Atom& atom = atoms[i];

    const double c = 0.5 * atom.inv_mass * dt;

    atom.velocity.x += (atom.force.x + atom.scratch.x) * c;
    atom.velocity.y += (atom.force.y + atom.scratch.y) * c;
    atom.velocity.z += (atom.force.z + atom.scratch.z) * c;
}

unsigned long long get_device_neighbour_count(Configuration& configuration)
{
    if (configuration.get_d_neighbour_list_size() == nullptr) {
        return 0;
    }

    unsigned long long n_pairs = 0;

    cuda_check(
        cudaMemcpy(&n_pairs,
                   configuration.get_d_neighbour_list_size(),
                   sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy neighbour-list size device to host"
    );

    if (configuration.get_max_nl_size() > 0) {
        n_pairs = std::min(
            n_pairs,
            static_cast<unsigned long long>(configuration.get_max_nl_size())
        );
    }

    return n_pairs;
}


// Main Verlet step

void VerletEngine::verlet_step(Configuration& configuration) {
    calculate_force(configuration);
    calculate_position(configuration);
    ConfigurationEngine::update_neighbour_list(configuration);
    calculate_velocity(configuration);
}

void VerletEngine::calculate_force(Configuration& configuration)
{
    auto t0 = std::chrono::steady_clock::now();
    auto& timer = TimerOnce::get();

    const std::size_t n_atoms = configuration._atoms.size();

    if (n_atoms == 0) {
        return;
    }

    if (configuration._d_atoms == nullptr) {
        ConfigurationEngine::upload_to_device(configuration);
    }

    if (configuration._d_neighbour_list == nullptr ||
        configuration._d_neighbour_list_size == nullptr) {
        ConfigurationEngine::make_neighbour_list(configuration);
    }

    const unsigned long long n_pairs_raw = get_device_neighbour_count(configuration);
    const std::size_t n_pairs = static_cast<std::size_t>(n_pairs_raw);

    const int threads = 256;

    const int atom_blocks =
        static_cast<int>((n_atoms + threads - 1) / threads);

    zero_forces_kernel<<<atom_blocks, threads>>>(
        configuration._d_atoms,
        n_atoms
    );

    cuda_check(cudaGetLastError(), "launch zero_forces_kernel");
    cuda_check(cudaDeviceSynchronize(), "zero_forces_kernel synchronize");

    if (n_pairs > 0) {
        const int pair_blocks =
            static_cast<int>((n_pairs + threads - 1) / threads);

        calculate_force_kernel<<<pair_blocks, threads>>>(
            configuration._d_atoms,
            configuration._d_neighbour_list,
            n_pairs,
            configuration._r_cutoff
        );

        cuda_check(cudaGetLastError(), "launch calculate_force_kernel");
        cuda_check(cudaDeviceSynchronize(), "calculate_force_kernel synchronize");
    }

    auto t1 = std::chrono::steady_clock::now();
    timer.update_force_calculations(t1 - t0);
}

void VerletEngine::calculate_position(Configuration& configuration)
{
    const std::size_t n_atoms = configuration._atoms.size();

    if (n_atoms == 0) {
        return;
    }

    if (configuration._d_atoms == nullptr) {
        ConfigurationEngine::upload_to_device(configuration);
    }

    const double dt = configuration._dt;

    const int threads = 256;
    const int blocks = static_cast<int>((n_atoms + threads - 1) / threads);

    calculate_position_kernel<<<blocks, threads>>>(
        configuration._d_atoms,
        n_atoms,
        dt
    );

    cuda_check(cudaGetLastError(), "launch calculate_position_kernel");
    cuda_check(cudaDeviceSynchronize(), "calculate_position_kernel synchronize");
}

void VerletEngine::calculate_velocity(Configuration& configuration)
{
    const std::size_t n_atoms = configuration._atoms.size();

    if (n_atoms == 0) {
        return;
    }

    if (configuration._d_atoms == nullptr) {
        ConfigurationEngine::upload_to_device(configuration);
    }

    const double dt = configuration._dt;

    const int threads = 256;
    const int blocks = static_cast<int>((n_atoms + threads - 1) / threads);

    store_force_kernel<<<blocks, threads>>>(
        configuration._d_atoms,
        n_atoms
    );

    cuda_check(cudaGetLastError(), "launch store_force_kernel");
    cuda_check(cudaDeviceSynchronize(), "store_force_kernel synchronize");

    calculate_force(configuration);

    calculate_velocity_kernel<<<blocks, threads>>>(
        configuration._d_atoms,
        n_atoms,
        dt
    );

    cuda_check(cudaGetLastError(), "launch calculate_velocity_kernel");
    cuda_check(cudaDeviceSynchronize(), "calculate_velocity_kernel synchronize");
}

/*
void VerletEngine::print_first_atom_device(const char* label, Configuration& configuration)
{
    Atom atom{};

    cuda_check(
        cudaMemcpy(&atom,
                   configuration._d_atoms,
                   sizeof(Atom),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy first atom device to host"
    );

    std::cout << label << '\n';
    std::cout << "  position = "
              << atom.position.x << ", "
              << atom.position.y << ", "
              << atom.position.z << '\n';

    std::cout << "  velocity = "
              << atom.velocity.x << ", "
              << atom.velocity.y << ", "
              << atom.velocity.z << '\n';

    std::cout << "  force    = "
              << atom.force.x << ", "
              << atom.force.y << ", "
              << atom.force.z << '\n';
}*/

}  // namespace