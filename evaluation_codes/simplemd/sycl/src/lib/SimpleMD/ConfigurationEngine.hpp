#ifndef CONFIGURATION_ENGINE_HPP
#define CONFIGURATION_ENGINE_HPP


/*********************************************************************************************************************************/

#include <random>
#include <numeric>
#include <omp.h>
#include <sycl/sycl.hpp>


#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "Atom.hpp"
#include "Configuration.hpp"
#include "Morse.hpp"
/*********************************************************************************************************************************/
namespace SimpleMD
{


class ConfigurationEngine
{

public:


    /**
     * Heat the configuration to a target temperature for testing.
     */
    static inline void heat(
                    Configuration& configuration, 
                    double T_target,
                    double kB = 8.617333262145e-5, // eV/K (change if you use SI)
                    std::optional<unsigned> seed = std::nullopt)
    {

        auto& atoms = configuration.get_atoms();
        const double sigma = configuration.get_heat();
        
        if (atoms.empty()) return;

        // RNG setup
        std::mt19937 rng(seed ? *seed : std::random_device{}());
        auto normal01 = std::normal_distribution<double>(0.0, 1.0);

        for (auto &atom : atoms) 
        {
            Maths::Vec3 d_position = {
                sigma * normal01(rng),
                sigma * normal01(rng),
                sigma * normal01(rng)
            };
            
            atom.set_position(atom.position + d_position);
        }
    }

    static inline void perturb(
                    const std::size_t n, 
                    const double sigma,
                    Configuration& configuration)
    {

        auto& atoms = configuration.get_atoms();
        
        if (atoms.empty()) return;

        // RNG setup
        std::mt19937 rng(42);
        auto normal01 = std::normal_distribution<double>(0.0, 1.0);

        // 1) Draw Maxwell–Boltzmann velocities: each component has variance kB*T/m
        for (auto &atom : atoms) 
        {

            const double new_sigma = sigma * ( 1.0 / n);
            Maths::Vec3 d_position = {
                new_sigma * normal01(rng),
                new_sigma * normal01(rng),
                new_sigma * normal01(rng)
            };
            
            atom.set_position(atom.position + d_position);
        }
    }


    static void make_neighbour_list(Configuration& configuration)
    {
        auto t0 = std::chrono::steady_clock::now();

        const auto& atoms = configuration.get_atoms();
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();

        auto& neighbour_list = configuration.get_neighbour_list();
        auto& timer = TimerOnce::get();

        std::size_t max_list_size = 1'000'000;

        neighbour_list.clear();
        neighbour_list.reserve(max_list_size);

        const double r_verlet_cutoff = configuration.get_r_verlet_cutoff();
        const double r_verlet_cutoff_sq = r_verlet_cutoff * r_verlet_cutoff;

        const std::size_t n_atoms = atoms.size();

        #pragma omp parallel
        {

            std::size_t max_local_list_size = max_list_size / omp_get_max_threads();

            std::vector<AtomPair> local_neighbour_list;
            local_neighbour_list.reserve(max_local_list_size);

            #pragma omp for schedule(static)
            for(size_t i=0; i<atoms.size()-1; ++i)
            {
                const auto& atom_i = atoms[i];

                for(size_t j=i+1; j<atoms.size(); ++j)
                {
                    const auto& atom_j = atoms[j];

                    // If a pair, add to local_neighbour_list
                    pair_atoms(configuration, r_verlet_cutoff_sq, atom_i, atom_j, i, j, local_neighbour_list);

                }
            }

            #pragma omp critical
            {
                neighbour_list.insert(neighbour_list.end(),
                                    local_neighbour_list.begin(), local_neighbour_list.end());
            }
        }
       
        std::cout << neighbour_list.size() << std::endl;
        auto t1 = std::chrono::steady_clock::now();
        timer.update_making_neighbour_list(t1 - t0);
    }


    static void pair_atoms(     Configuration& configuration,
                                const double r_verlet_cutoff_sq, 
                                const Atom& atom_i, 
                                const Atom& atom_j, 
                                size_t ni, 
                                size_t nj,
                                std::vector<AtomPair>& local_neighbour_list
                            )
    {
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();


        for(int i = -1; i <= 1; i++)
        {        
            for(int j = -1; j <= 1; j++)
            {    
                for(int k = -1; k <= 1; k++)
                {
                    Maths::Vec3 position_offset = {1.0 * i, 1.0 * j, 1.0 * k};

                    // Get squared separation between the two
                    const auto r_vec = Maths::Vec3::separation( 
                                                                alat, 
                                                                basis, 
                                                                atom_i.position, 
                                                                atom_j.position + position_offset
                                                            );

                    auto r_sq = r_vec.length_squared();

                    if(r_sq <= r_verlet_cutoff_sq)
                    {

                        SimpleMD::AtomPair atom_pair {};

                        atom_pair.atom_i_idx = ni;
                        atom_pair.atom_j_idx = nj;

                        atom_pair.r = std::sqrt(r_sq);
                        atom_pair.r = std::max(atom_pair.r, 1.0e-9);

                        auto inv_r = 1.0 / atom_pair.r;
                        atom_pair.u_vec = inv_r * r_vec;
                        atom_pair.position_offset = position_offset;

                        local_neighbour_list.emplace_back(atom_pair);                     

                    }
                }
            }

        }

    }


    static void update_neighbour_list(Configuration& configuration)
    {
        const auto& atoms = configuration.get_atoms();
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();

        auto& neighbour_list = configuration.get_neighbour_list();     
        auto& timer = TimerOnce::get();   

        const std::size_t n_pairs = neighbour_list.size();

        auto t0 = std::chrono::steady_clock::now();

        #pragma omp parallel for
        for (std::size_t  idx = 0; idx < n_pairs; ++idx)
        {
            auto& atom_pair = neighbour_list[idx];

            // Load atoms and ghost atom
            const auto& atom_i = atoms[atom_pair.atom_i_idx];
            const auto& atom_j = atoms[atom_pair.atom_j_idx];
            
            // Get squared separation between the two
            const auto r_vec = Maths::Vec3::separation( 
                                                        alat, 
                                                        basis, 
                                                        atom_i.position, 
                                                        atom_j.position + atom_pair.position_offset
                                                    );
            auto r_sq = r_vec.length_squared();

            // Update r and u_vec
            atom_pair.r = std::sqrt(r_sq);

            if (atom_pair.r > 0.0)
            {
                atom_pair.u_vec = (1.0 / atom_pair.r) * r_vec;
            }
            else
            {
                atom_pair.u_vec = {};
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        timer.update_updating_neighbour_list(t1 - t0);

    }



    static void record_to_xyz(const int time_step,
                            const std::filesystem::path& xyz_file,
                            Configuration& configuration)
    {
        const auto& atoms = configuration.get_atoms();
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();

        // Ensure the directory exists
        if (xyz_file.has_parent_path()) 
        {
            std::error_code ec;
            std::filesystem::create_directories(xyz_file.parent_path(), ec);
        }

        // Only truncate the first time (or if file doesn't exist)
        static bool first_call = true;
        std::ios_base::openmode mode =
            (first_call || !std::filesystem::exists(xyz_file))
            ? (std::ios::out | std::ios::trunc)
            : (std::ios::out | std::ios::app);

        std::ofstream out(xyz_file, mode);
        if (!out)
        {
            throw std::runtime_error("record_to_xyz: failed to open file: " + xyz_file.string());
        }

        out << atoms.size() << '\n';

        // Write lattice vectors and metadata in ASE/OVITO-compatible format
        out << std::fixed << std::setprecision(6);
        out << "Lattice=\""
            << basis[0] * alat << " " << basis[1] * alat << " " << basis[2] * alat << "  "
            << basis[3] * alat << " " << basis[4] * alat << " " << basis[5] * alat << "  "
            << basis[6] * alat << " " << basis[7] * alat << " " << basis[8] * alat << "\" "
            << "Properties=species:S:1:pos:R:3 timestep=" << time_step
            << " alat=" << alat << '\n';

        // Convert positions from fractional → Cartesian and write atoms
        out << std::setprecision(10);
        for (const auto& atom : atoms)
        {
            const auto& f = atom.position; // fractional
            // Cartesian = alat * (basis * fractional)
            const Maths::Vec3 cart = alat * (basis * f);

            // Replace "X" with element symbol if available
            out << "X " << cart.x << ' ' << cart.y << ' ' << cart.z << '\n';
        }

        out.flush();
        first_call = false;
    }





    //     SYCL additions
    // #############################


    static void upload_to_device(Configuration& configuration)
    {
        std::cout << "Loading to device." << std::endl;
        // Existing SYCL queue owned by the configuration
        sycl::queue& q = configuration.q;

        // ---- atoms: allocate on device (USM) and memcpy ----
        const std::size_t n = configuration._atoms.size();

        if (n == 0) 
        {
            return;
        }

        if (configuration._d_atoms != nullptr)
        {
            sycl::free(configuration._d_atoms, q);
        }

        // Allocate size
        configuration._d_atoms = sycl::malloc_device<Atom>(n, q);


        // Copy host atoms to device
        q.memcpy(configuration._d_atoms, configuration._atoms.data(), n * sizeof(Atom)).wait();

              
        std::cout << "Loaded." << std::endl;  
    }



    static void download_from_device(Configuration& configuration)
    {
        sycl::queue& q = configuration.q;
        auto& atoms    = configuration._atoms;

        const std::size_t n = atoms.size();

        if (n == 0)
            return;

        if (configuration._d_atoms == nullptr)
        {
            // Nothing on device to download
            return;
        }

        // Copy device atoms back to host atoms
        q.memcpy(atoms.data(), configuration._d_atoms, n * sizeof(Atom)).wait();
    }



    static void make_neighbour_list_sycl(Configuration& configuration)
    {
        auto t0 = std::chrono::steady_clock::now();

        auto& timer          = TimerOnce::get();
        auto& neighbour_list = configuration._neighbour_list; // friend access
        sycl::queue& q       = configuration.q;

        const std::size_t n_atoms = configuration._atoms.size();
        if (n_atoms == 0) 
            return;

        // Host-side capacity we want on device
        const std::size_t max_list_size = 1'000'000;

        // ------------------------------------------------------------------
        // Allocate / grow device neighbour list buffer
        // ------------------------------------------------------------------
        if (configuration._d_neighbour_list == nullptr)
        {
            configuration._d_neighbour_list = sycl::malloc_device<AtomPair>(max_list_size, q);
        }

        Atom* d_atoms  = configuration._d_atoms;
        AtomPair* d_neigh = configuration._d_neighbour_list;

        // ------------------------------------------------------------------
        // Allocate device-visible counter (shared USM) if needed
        // ------------------------------------------------------------------
        if (configuration._d_neighbour_list_size == nullptr)
        {
            configuration._d_neighbour_list_size =
                sycl::malloc_shared<std::size_t>(1, q);
        }

        // Reset counter to zero before kernel
        *configuration._d_neighbour_list_size = 0;

        // Capture simulation parameters from host
        double d_alat = configuration._alat;
        auto d_basis = configuration._basis;
        double d_r_verlet_cutoff = configuration._r_verlet_cutoff;
        double d_r_verlet_cutoff_sq = d_r_verlet_cutoff * d_r_verlet_cutoff;

        std::size_t* d_count = configuration._d_neighbour_list_size;

        // ------------------------------------------------------------------
        // Launch kernel: parallel over (i,j) pairs, skipping j <= i
        // ------------------------------------------------------------------
        q.submit([&](sycl::handler& h)
        {
            h.parallel_for(
                sycl::range<2>(n_atoms, n_atoms),
                [=](sycl::id<2> idx)
                {
                    const std::size_t i = idx[0];
                    const std::size_t j = idx[1];

                    if (j <= i)
                        return; // only i < j

                    const Atom& atom_i = d_atoms[i];
                    const Atom& atom_j = d_atoms[j];

                    // Loop over periodic images (-1,0,1) in each direction
                    for (int ii = -1; ii <= 1; ++ii)
                    {
                        for (int jj = -1; jj <= 1; ++jj)
                        {
                            for (int kk = -1; kk <= 1; ++kk)
                            {
                                Maths::Vec3 position_offset = { 1.0 * ii, 1.0 * jj, 1.0 * kk };

                                const auto r_vec = Maths::Vec3::separation(
                                    d_alat,
                                    d_basis,
                                    atom_i.position,
                                    atom_j.position + position_offset
                                );

                                const double r_sq = r_vec.length_squared();
                                if (r_sq > d_r_verlet_cutoff_sq)
                                {
                                    continue;
                                }

                                const double r = sycl::max(sycl::sqrt(r_sq), 1.0e-9);
                                if (r <= 0.0)
                                {
                                    continue;
                                }

                                // Atomic increment of global neighbour counter
                                sycl::atomic_ref<
                                    std::size_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space
                                > counter(*d_count);

                                const std::size_t idx_out = counter.fetch_add(1);

                                // Guard against overflow of buffer
                                if (idx_out >= max_list_size)
                                {
                                    return;
                                }

                                AtomPair pair;
                                pair.atom_i_idx = i;
                                pair.atom_j_idx = j;
                                pair.r = r;
                                pair.position_offset = position_offset;
                                pair.u_vec = (1.0 / r) * r_vec;

                                d_neigh[idx_out] = pair;
                            }
                        }
                    }
                });
        }).wait();

        // ------------------------------------------------------------------
        // Use the neighbour count on host
        // (malloc_shared => no memcpy required)
        // ------------------------------------------------------------------
        std::size_t neighbour_count = *configuration._d_neighbour_list_size;

        // If you want the host vector to mirror the device list:
        neighbour_list.resize(neighbour_count);
        if (neighbour_count > 0)
        {
            q.memcpy(neighbour_list.data(), configuration._d_neighbour_list, neighbour_count * sizeof(AtomPair)).wait();
        }

        std::cout << "Device neighbour count: " << neighbour_count << "\n";

        auto t1 = std::chrono::steady_clock::now();
        timer.update_making_neighbour_list(t1 - t0);
    }



    static void update_neighbour_list_sycl(Configuration& configuration)
    {
        auto t0    = std::chrono::steady_clock::now();
        auto& timer = TimerOnce::get();

        sycl::queue& q = configuration.q;

        Atom* d_atoms = configuration._d_atoms;
        AtomPair* d_neigh = configuration._d_neighbour_list;

        // number of pairs currently in neighbour list (device-visible counter)
        std::size_t n_pairs = 0;
        if (configuration._d_neighbour_list_size != nullptr)
        {
            n_pairs = *configuration._d_neighbour_list_size;  // malloc_shared → directly readable
        }

        if (n_pairs == 0)
        {
            auto t1 = std::chrono::steady_clock::now();
            timer.update_updating_neighbour_list(t1 - t0);
            return;
        }

        // capture host parameters by value (copied as kernel arguments)
        const double alat = configuration.get_alat();
        const auto   basis = configuration.get_basis();  // std::array<double, 9>

        q.submit([&](sycl::handler& h)
        {
            h.parallel_for(
                sycl::range<1>(n_pairs),
                [=](sycl::id<1> idx)
                {
                    const std::size_t k = idx[0];

                    AtomPair& atom_pair = d_neigh[k];

                    const Atom& atom_i = d_atoms[atom_pair.atom_i_idx];
                    const Atom& atom_j = d_atoms[atom_pair.atom_j_idx];

                    // recompute separation with current positions and stored offset
                    const auto r_vec = Maths::Vec3::separation(
                        alat,
                        basis,
                        atom_i.position,
                        atom_j.position + atom_pair.position_offset
                    );

                    const double r_sq = r_vec.length_squared();
                    const double r    = sycl::sqrt(r_sq);

                    atom_pair.r = r;

                    if (r > 0.0)
                    {
                        atom_pair.u_vec = (1.0 / r) * r_vec;
                    }
                    else
                    {
                        atom_pair.u_vec = {}; // zero vector
                    }
                });
        }).wait();

        auto t1 = std::chrono::steady_clock::now();
        timer.update_updating_neighbour_list(t1 - t0);
    }





    static void record_to_xyz_sycl(const int time_step,
                            const std::filesystem::path& xyz_file,
                            Configuration& configuration)
    {

        download_from_device(configuration);

        const auto& atoms = configuration.get_atoms();
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();

        // Ensure the directory exists
        if (xyz_file.has_parent_path()) 
        {
            std::error_code ec;
            std::filesystem::create_directories(xyz_file.parent_path(), ec);
        }

        // Only truncate the first time (or if file doesn't exist)
        static bool first_call = true;
        std::ios_base::openmode mode =
            (first_call || !std::filesystem::exists(xyz_file))
            ? (std::ios::out | std::ios::trunc)
            : (std::ios::out | std::ios::app);

        std::ofstream out(xyz_file, mode);
        if (!out)
        {
            throw std::runtime_error("record_to_xyz: failed to open file: " + xyz_file.string());
        }

        out << atoms.size() << '\n';

        // Write lattice vectors and metadata in ASE/OVITO-compatible format
        out << std::fixed << std::setprecision(6);
        out << "Lattice=\""
            << basis[0] * alat << " " << basis[1] * alat << " " << basis[2] * alat << "  "
            << basis[3] * alat << " " << basis[4] * alat << " " << basis[5] * alat << "  "
            << basis[6] * alat << " " << basis[7] * alat << " " << basis[8] * alat << "\" "
            << "Properties=species:S:1:pos:R:3 timestep=" << time_step
            << " alat=" << alat << '\n';

        // Convert positions from fractional → Cartesian and write atoms
        out << std::setprecision(10);
        for (const auto& atom : atoms)
        {
            const auto& f = atom.position; // fractional
            // Cartesian = alat * (basis * fractional)
            const Maths::Vec3 cart = alat * (basis * f);

            // Replace "X" with element symbol if available
            out << "X " << cart.x << ' ' << cart.y << ' ' << cart.z << '\n';
        }

        out.flush();
        first_call = false;
    }



};

}

#endif