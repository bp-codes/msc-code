#ifndef VERLET_ENGINE_HPP
#define VERLET_ENGINE_HPP


/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "Atom.hpp"
#include "Configuration.hpp"
#include "ConfigurationEngine.hpp"
#include "Morse.hpp"
/*********************************************************************************************************************************/
namespace SimpleMD
{


class VerletEngine
{

public:


    //     SYCL 
    // #############################

    static void calculate_force_sycl(Configuration& configuration)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto& timer = TimerOnce::get();
        sycl::queue& q = configuration.q;

        auto& atoms = configuration._atoms;   // friend access
        const std::size_t n_atoms = atoms.size();
        if (n_atoms == 0)
        {
            return;
        }

        // Device data
        Atom* d_atoms  = configuration._d_atoms;
        AtomPair* d_neigh  = configuration._d_neighbour_list;
        std::size_t n_pairs = 0;

        if (configuration._d_neighbour_list_size != nullptr)
        {
            n_pairs = *configuration._d_neighbour_list_size;
        }
        else
        {
            n_pairs = 0;
        }

        const double r_cutoff = configuration._r_cutoff;

        // zero forces
        q.submit([&](sycl::handler& h)
        {
            h.parallel_for(
                sycl::range<1>(n_atoms),
                [=](sycl::id<1> idx)
                {
                    const std::size_t i = idx[0];
                    d_atoms[i].force = {0.0, 0.0, 0.0};
                });
        }).wait();

        // compute forces
        q.submit([&](sycl::handler& h)
        {
            h.parallel_for(
                sycl::range<1>(n_pairs),
                [=](sycl::id<1> idx)
                {
                    const std::size_t k = idx[0];
                    const AtomPair pair = d_neigh[k];

                    if (pair.r > r_cutoff)
                        return;

                    // Morse force magnitude
                    const double force_mag =
                        SimpleMD::Morse::force(pair.r, 0.343, 1.44, 2.863);

                    const auto vforce = force_mag * pair.u_vec; // Vec3

                    const std::size_t i = pair.atom_i_idx;
                    const std::size_t j = pair.atom_j_idx;

                    // Atom references on device
                    auto& fi = d_atoms[i].force;
                    auto& fj = d_atoms[j].force;

                    // Adjust component access (x/y/z or [0]/[1]/[2]) to your Vec3
                    using atomic_double = sycl::atomic_ref<
                        double,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>;

                    // i atom gets -vforce
                    atomic_double(fi.x).fetch_add(-vforce.x);
                    atomic_double(fi.y).fetch_add(-vforce.y);
                    atomic_double(fi.z).fetch_add(-vforce.z);

                    // j atom gets +vforce
                    atomic_double(fj.x).fetch_add(vforce.x);
                    atomic_double(fj.y).fetch_add(vforce.y);
                    atomic_double(fj.z).fetch_add(vforce.z);
                });
        }).wait();


        auto t1 = std::chrono::steady_clock::now();
        timer.update_force_calculations(t1 - t0);
    }


    static void calculate_position_sycl(SimpleMD::Configuration& configuration)
    {
        auto& q      = configuration.q;
        auto& atoms  = configuration.get_atoms();
        const std::size_t n_atoms = atoms.size();
        if (n_atoms == 0) return;

        const double dt = configuration.get_dt();

        Atom* d_atoms = configuration._d_atoms;

        // Update positions on device
        q.submit([&](sycl::handler& h)
        {
            h.parallel_for(
                sycl::range<1>(n_atoms),
                [=](sycl::id<1> idx)
                {
                    const std::size_t i = idx[0];
                    Atom& atom = d_atoms[i];

                    atom.position += atom.velocity * dt + 0.5 * atom.force * atom.inv_mass * dt * dt;

                    // Assuming this is SYCL-device safe
                    atom.position.unit_cell_pbc();
                });
        }).wait();

    }


    static void calculate_velocity_sycl(SimpleMD::Configuration& configuration)
    {
        auto& q      = configuration.q;
        auto& atoms  = configuration.get_atoms();
        const std::size_t n_atoms = atoms.size();
        if (n_atoms == 0) return;

        const double dt = configuration.get_dt();
        Atom* d_atoms = configuration._d_atoms;

        // 1. Store starting force in scratch on device
        q.submit([&](sycl::handler& h)
        {
            h.parallel_for(
                sycl::range<1>(n_atoms),
                [=](sycl::id<1> idx)
                {
                    const std::size_t i = idx[0];
                    d_atoms[i].scratch = d_atoms[i].force;
                });
        }).wait();

        // 2. Recompute forces on device
        calculate_force_sycl(configuration);

        // 3. Update velocities using old (scratch) + new (force)
        q.submit([&](sycl::handler& h)
        {
            h.parallel_for(
                sycl::range<1>(n_atoms),
                [=](sycl::id<1> idx)
                {
                    const std::size_t i = idx[0];
                    Atom& atom = d_atoms[i];

                    atom.velocity += 0.5 * (atom.force + atom.scratch)
                                    * atom.inv_mass * dt;
                });
        }).wait();

    }




    static void vertlet_step_sycl(Configuration& configuration)
    {

        calculate_force_sycl(configuration);
        calculate_position_sycl(configuration);
        ConfigurationEngine::update_neighbour_list_sycl(configuration);
        calculate_velocity_sycl(configuration);

    }

};

}

#endif