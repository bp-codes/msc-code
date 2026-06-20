
#include "SimpleMD/Atom.hpp"
#include "SimpleMD/AtomPair.hpp"
#include "Maths/Vec3.hpp"
#include "SimpleMD/ConfigurationEngine.hpp"

namespace SimpleMD {

inline void ConfigurationEngine::cuda_check(cudaError_t err, const std::string& message)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(err));
    }
}

struct Basis3 {
    double v[9];
};

// CUDA kernels

__global__ void make_neighbour_list_kernel(const Atom* atoms,
                                           AtomPair* neighbour_list,
                                           unsigned long long* neighbour_count,
                                           std::size_t n_atoms,
                                           unsigned long long max_list_size,
                                           double alat,
                                           Basis3 basis,
                                           double r_verlet_cutoff_sq)
{
    const std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_atoms || j >= n_atoms) {
        return;
    }

    if (j <= i) {
        return;
    }

    const Atom& atom_i = atoms[i];
    const Atom& atom_j = atoms[j];

    for (int ii = -1; ii <= 1; ++ii) {
        for (int jj = -1; jj <= 1; ++jj) {
            for (int kk = -1; kk <= 1; ++kk) {
                const Maths::Vec3 position_offset = {
                    static_cast<double>(ii),
                    static_cast<double>(jj),
                    static_cast<double>(kk)
                };

                // Expanded, not using Vec3 method
                const double dx = atom_j.position.x + static_cast<double>(ii) - atom_i.position.x;
                const double dy = atom_j.position.y + static_cast<double>(jj) - atom_i.position.y;
                const double dz = atom_j.position.z + static_cast<double>(kk) - atom_i.position.z;

                const double rx = alat * (
                    basis.v[0] * dx +
                    basis.v[1] * dy +
                    basis.v[2] * dz
                );

                const double ry = alat * (
                    basis.v[3] * dx +
                    basis.v[4] * dy +
                    basis.v[5] * dz
                );

                const double rz = alat * (
                    basis.v[6] * dx +
                    basis.v[7] * dy +
                    basis.v[8] * dz
                );

                const double r_sq = rx * rx + ry * ry + rz * rz;

                if (r_sq > r_verlet_cutoff_sq) {
                    continue;
                }

                const double r = fmax(sqrt(r_sq), 1.0e-9);

                const unsigned long long idx_out = atomicAdd(neighbour_count, 1ULL);

                if (idx_out >= max_list_size) {
                    return;
                }

                AtomPair pair;
                pair.atom_i_idx = i;
                pair.atom_j_idx = j;
                pair.r = r;
                pair.position_offset = Maths::Vec3{
                    static_cast<double>(ii),
                    static_cast<double>(jj),
                    static_cast<double>(kk)
                };
                pair.u_vec = Maths::Vec3{
                    rx / r,
                    ry / r,
                    rz / r
                };

                neighbour_list[idx_out] = pair;
            }
        }
    }
}

__global__ void update_neighbour_list_kernel(const Atom* atoms,
                                             AtomPair* neighbour_list,
                                             std::size_t n_pairs,
                                             double alat,
                                             Basis3 basis)
{
    const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= n_pairs) {
        return;
    }

    AtomPair& atom_pair = neighbour_list[k];

    const Atom& atom_i = atoms[atom_pair.atom_i_idx];
    const Atom& atom_j = atoms[atom_pair.atom_j_idx];

    const double dx = atom_j.position.x + atom_pair.position_offset.x - atom_i.position.x;
    const double dy = atom_j.position.y + atom_pair.position_offset.y - atom_i.position.y;
    const double dz = atom_j.position.z + atom_pair.position_offset.z - atom_i.position.z;

    const double rx = alat * (
        basis.v[0] * dx +
        basis.v[1] * dy +
        basis.v[2] * dz
    );

    const double ry = alat * (
        basis.v[3] * dx +
        basis.v[4] * dy +
        basis.v[5] * dz
    );

    const double rz = alat * (
        basis.v[6] * dx +
        basis.v[7] * dy +
        basis.v[8] * dz
    );

    const double r_sq = rx * rx + ry * ry + rz * rz;
    const double r = sqrt(r_sq);

    atom_pair.r = r;

    if (r > 0.0) {
        atom_pair.u_vec.x = rx / r;
        atom_pair.u_vec.y = ry / r;
        atom_pair.u_vec.z = rz / r;
    } else {
        atom_pair.u_vec.x = 0.0;
        atom_pair.u_vec.y = 0.0;
        atom_pair.u_vec.z = 0.0;
    }
}

   
//     CUDA modifications
// #############################

void ConfigurationEngine::upload_to_device(Configuration& configuration) {
    std::cout << "Loading atoms to CUDA device.\n";

    const std::size_t n = configuration._atoms.size();

    if (n == 0) {
        return;
    }

    if (configuration._d_atoms != nullptr) {
        cuda_check(cudaFree(configuration._d_atoms), "cudaFree _d_atoms");
        configuration._d_atoms = nullptr;
    }

    cuda_check(
        cudaMalloc(&configuration._d_atoms, n * sizeof(Atom)),
        "cudaMalloc _d_atoms"
    );

    cuda_check(
        cudaMemcpy(configuration._d_atoms,
                    configuration._atoms.data(),
                    n * sizeof(Atom),
                    cudaMemcpyHostToDevice),
        "cudaMemcpy atoms host to device"
    );

    std::cout << "Loaded atoms to CUDA device.\n";
}

void ConfigurationEngine::download_from_device(Configuration& configuration) {
    auto& atoms = configuration._atoms;
    const std::size_t n = atoms.size();

    if (n == 0) {
        return;
    }

    if (configuration._d_atoms == nullptr) {
        return;
    }

    cuda_check(
        cudaMemcpy(atoms.data(),
                    configuration._d_atoms,
                    n * sizeof(Atom),
                    cudaMemcpyDeviceToHost),
        "cudaMemcpy atoms device to host"
    );
}

void ConfigurationEngine::free_device(Configuration& configuration) {
    if (configuration._d_atoms != nullptr) {
        cuda_check(cudaFree(configuration._d_atoms), "cudaFree _d_atoms");
        configuration._d_atoms = nullptr;
    }

    if (configuration._d_neighbour_list != nullptr) {
        cuda_check(cudaFree(configuration._d_neighbour_list), "cudaFree _d_neighbour_list");
        configuration._d_neighbour_list = nullptr;
    }

    if (configuration._d_neighbour_list_size != nullptr) {
        cuda_check(cudaFree(configuration._d_neighbour_list_size),
                    "cudaFree _d_neighbour_list_size");
        configuration._d_neighbour_list_size = nullptr;
    }
}

void ConfigurationEngine::make_neighbour_list(Configuration& configuration)
{
    auto t0 = std::chrono::steady_clock::now();

    auto& timer = TimerOnce::get();
    auto& neighbour_list = configuration._neighbour_list;

    const std::size_t n_atoms = configuration._atoms.size();

    if (n_atoms == 0) {
        return;
    }

    // Upload configuration if not already done
    if (configuration._d_atoms == nullptr) {
        upload_to_device(configuration);
    }

    const unsigned long long max_list_size =
        configuration._max_nl_size > 0
            ? static_cast<unsigned long long>(configuration._max_nl_size)
            : 8'000'000ULL;

    if (configuration._d_neighbour_list == nullptr) {
        cuda_check(
            cudaMalloc(&configuration._d_neighbour_list,
                    max_list_size * sizeof(AtomPair)),
            "cudaMalloc _d_neighbour_list"
        );
    }

    if (configuration._d_neighbour_list_size == nullptr) {
        cuda_check(
            cudaMalloc(&configuration._d_neighbour_list_size,
                    sizeof(unsigned long long)),
            "cudaMalloc _d_neighbour_list_size"
        );
    }

    cuda_check(
        cudaMemset(configuration._d_neighbour_list_size,
                0,
                sizeof(unsigned long long)),
        "cudaMemset _d_neighbour_list_size"
    );

    Atom* d_atoms = configuration._d_atoms;
    AtomPair* d_neigh = configuration._d_neighbour_list;
    unsigned long long* d_count = configuration._d_neighbour_list_size;

    const double alat = configuration._alat;
    Basis3 basis {};

    for (std::size_t i = 0; i < 9; ++i) {
        basis.v[i] = configuration._basis[i];
    }

    const double r_verlet_cutoff = configuration._r_verlet_cutoff;
    const double r_verlet_cutoff_sq = r_verlet_cutoff * r_verlet_cutoff;

    const dim3 threads(16, 16);
    const dim3 blocks(
        static_cast<unsigned int>((n_atoms + threads.x - 1) / threads.x),
        static_cast<unsigned int>((n_atoms + threads.y - 1) / threads.y)
    );

    make_neighbour_list_kernel<<<blocks, threads>>>(
        d_atoms,
        d_neigh,
        d_count,
        n_atoms,
        max_list_size,
        alat,
        basis,
        r_verlet_cutoff_sq
    );

    cuda_check(cudaGetLastError(), "launch make_neighbour_list_kernel");
    cuda_check(cudaDeviceSynchronize(), "make_neighbour_list_kernel synchronize");

    unsigned long long neighbour_count_raw = 0;

    cuda_check(
        cudaMemcpy(&neighbour_count_raw,
                configuration._d_neighbour_list_size,
                sizeof(unsigned long long),
                cudaMemcpyDeviceToHost),
        "cudaMemcpy neighbour count device to host"
    );

    const unsigned long long neighbour_count_capped =
        neighbour_count_raw > max_list_size ? max_list_size : neighbour_count_raw;

    if (neighbour_count_raw > max_list_size) {
        std::cerr << "Warning: neighbour list overflow. "
                << "Found " << neighbour_count_raw
                << " pairs, but capacity is only "
                << max_list_size << ".\n";
    }

    const std::size_t neighbour_count =
        static_cast<std::size_t>(neighbour_count_capped);

    neighbour_list.resize(neighbour_count);

    if (neighbour_count > 0) {
        cuda_check(
            cudaMemcpy(neighbour_list.data(),
                    configuration._d_neighbour_list,
                    neighbour_count * sizeof(AtomPair),
                    cudaMemcpyDeviceToHost),
            "cudaMemcpy neighbour list device to host"
        );
    }

    configuration._max_nl_size = static_cast<std::size_t>(max_list_size);

    std::cout << "Device neighbour count: "
            << neighbour_count_raw
            << '\n';

    auto t1 = std::chrono::steady_clock::now();
    timer.update_making_neighbour_list(t1 - t0);
}


void ConfigurationEngine::update_neighbour_list(Configuration& configuration)
{
    auto t0 = std::chrono::steady_clock::now();
    auto& timer = TimerOnce::get();

    if (configuration._d_atoms == nullptr) {
        return;
    }

    if (configuration._d_neighbour_list == nullptr) {
        return;
    }

    if (configuration._d_neighbour_list_size == nullptr) {
        return;
    }

    unsigned long long n_pairs_raw = 0;

    cuda_check(
        cudaMemcpy(&n_pairs_raw,
                   configuration._d_neighbour_list_size,
                   sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy neighbour-list size device to host"
    );

    const unsigned long long max_list_size =
        configuration._max_nl_size > 0
            ? static_cast<unsigned long long>(configuration._max_nl_size)
            : n_pairs_raw;

    const unsigned long long n_pairs_capped =
        n_pairs_raw > max_list_size ? max_list_size : n_pairs_raw;

    const std::size_t n_pairs =
        static_cast<std::size_t>(n_pairs_capped);

    if (n_pairs == 0) {
        auto t1 = std::chrono::steady_clock::now();
        timer.update_updating_neighbour_list(t1 - t0);
        return;
    }

    const double alat = configuration._alat;

    Basis3 basis{};

    for (std::size_t i = 0; i < 9; ++i) {
        basis.v[i] = configuration._basis[i];
    }

    const int threads = 256;
    const int blocks = static_cast<int>((n_pairs + threads - 1) / threads);

    update_neighbour_list_kernel<<<blocks, threads>>>(
        configuration._d_atoms,
        configuration._d_neighbour_list,
        n_pairs,
        alat,
        basis
    );

    cuda_check(cudaGetLastError(), "launch update_neighbour_list_kernel");
    cuda_check(cudaDeviceSynchronize(), "update_neighbour_list_kernel synchronize");

    auto t1 = std::chrono::steady_clock::now();
    timer.update_updating_neighbour_list(t1 - t0);
}


void ConfigurationEngine::record_to_xyz(const int time_step, Configuration& configuration)
{
    download_from_device(configuration);

    const auto& atoms = configuration.get_atoms();
    const double alat = configuration.get_alat();
    const auto& basis = configuration.get_basis();

    const auto xyz_file = configuration.get_output_dir() / "out.xyz";

    if (xyz_file.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(xyz_file.parent_path(), ec);

        if (ec) {
            throw std::runtime_error(
                "record_to_xyz: failed to create directory: " +
                xyz_file.parent_path().string() +
                ": " +
                ec.message()
            );
        }
    }

    static bool first_call = true;

    const std::ios_base::openmode mode =
        (first_call || !std::filesystem::exists(xyz_file))
            ? (std::ios::out | std::ios::trunc)
            : (std::ios::out | std::ios::app);

    std::ofstream out(xyz_file, mode);

    if (!out) {
        throw std::runtime_error(
            "record_to_xyz: failed to open file: " + xyz_file.string()
        );
    }

    out << atoms.size() << '\n';

    out << std::fixed << std::setprecision(6);

    out << "Lattice=\""
        << basis[0] * alat << ' '
        << basis[1] * alat << ' '
        << basis[2] * alat << "  "
        << basis[3] * alat << ' '
        << basis[4] * alat << ' '
        << basis[5] * alat << "  "
        << basis[6] * alat << ' '
        << basis[7] * alat << ' '
        << basis[8] * alat << "\" "
        << "Properties=species:S:1:pos:R:3 "
        << "timestep=" << time_step << ' '
        << "alat=" << alat
        << '\n';

    out << std::setprecision(10);

    for (const auto& atom : atoms) {
        const auto& f = atom.position;

        const double x = alat * (
            basis[0] * f.x +
            basis[1] * f.y +
            basis[2] * f.z
        );

        const double y = alat * (
            basis[3] * f.x +
            basis[4] * f.y +
            basis[5] * f.z
        );

        const double z = alat * (
            basis[6] * f.x +
            basis[7] * f.y +
            basis[8] * f.z
        );

        out << "X " << x << ' ' << y << ' ' << z << '\n';
    }

    out.flush();

    first_call = false;
}

}  // namespace SimpleMD