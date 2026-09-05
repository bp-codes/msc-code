#ifndef TIMER_HPP
#define TIMER_HPP

/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
/*********************************************************************************************************************************/

#include <chrono>    // std::chrono::duration, std::chrono::duration_cast
#include <iostream>  // std::cout
#include <string>    // std::string

/**
 * @brief Accumulates and reports timing information for simulation components.
 *
 * Tracks total simulation time, neighbour list construction/update time,
 * and force calculation time.
 */
class Timer {
private:
    std::chrono::duration<double> _overall_time{};
    std::chrono::duration<double> _force_calculations{};
    std::chrono::duration<double> _making_neighbour_list{};
    std::chrono::duration<double> _updating_neighbour_list{};

public:
    Timer() = default;

    /**
     * @brief Add elapsed time to the overall simulation timer.
     *
     * @tparam Rep Representation type of duration.
     * @tparam Period Period type of duration.
     * @param time_elapsed Duration to add.
     */
    template <typename Rep, typename Period>
    inline void update_overall_time(const std::chrono::duration<Rep, Period>& time_elapsed) {
        _overall_time += std::chrono::duration_cast<std::chrono::duration<double>>(time_elapsed);
    }

    /**
     * @brief Add elapsed time to the force calculation timer.
     *
     * @tparam Rep Representation type of duration.
     * @tparam Period Period type of duration.
     * @param time_elapsed Duration to add.
     */
    template <typename Rep, typename Period>
    inline void update_force_calculations(const std::chrono::duration<Rep, Period>& time_elapsed) {
        _force_calculations +=
            std::chrono::duration_cast<std::chrono::duration<double>>(time_elapsed);
    }

    /**
     * @brief Add elapsed time to the neighbour list construction timer.
     *
     * @tparam Rep Representation type of duration.
     * @tparam Period Period type of duration.
     * @param time_elapsed Duration to add.
     */
    template <typename Rep, typename Period>
    inline void update_making_neighbour_list(
        const std::chrono::duration<Rep, Period>& time_elapsed) {
        _making_neighbour_list +=
            std::chrono::duration_cast<std::chrono::duration<double>>(time_elapsed);
    }

    /**
     * @brief Add elapsed time to the neighbour list update timer.
     *
     * @tparam Rep Representation type of duration.
     * @tparam Period Period type of duration.
     * @param time_elapsed Duration to add.
     */
    template <typename Rep, typename Period>
    inline void update_updating_neighbour_list(
        const std::chrono::duration<Rep, Period>& time_elapsed) {
        _updating_neighbour_list +=
            std::chrono::duration_cast<std::chrono::duration<double>>(time_elapsed);
    }

    /**
     * @brief Reset all accumulated timers to zero.
     */
    void reset() noexcept {
        _overall_time = std::chrono::duration<double>::zero();
        _force_calculations = std::chrono::duration<double>::zero();
        _making_neighbour_list = std::chrono::duration<double>::zero();
        _updating_neighbour_list = std::chrono::duration<double>::zero();
    }

    /**
     * @brief Print timing summary to standard output.
     */
    void print_times() {
        std::cout << "Force calcs:      " << _force_calculations.count() << "\n";
        std::cout << "Making nl:        " << _making_neighbour_list.count() << "\n";
        std::cout << "Update nl:        " << _updating_neighbour_list.count() << "\n";
        std::cout << "Overall time:     " << _overall_time.count() << "\n";
    }

    inline double get_force_calculations_seconds() {
        return _force_calculations.count();
    }

    inline double get_making_neighbour_list_seconds() {
        return _making_neighbour_list.count();
    }

    inline double get_updating_neighbour_list_seconds() {
        return _updating_neighbour_list.count();
    }

    inline double get_overall_time_seconds() {
        return _overall_time.count();
    }
};

// Singleton instance accessor
SINGLETON(Timer);

#endif
