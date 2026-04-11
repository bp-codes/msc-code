#ifndef TIMER_HPP
#define TIMER_HPP

#include <iostream>
#include <chrono>
#include <string>
#include "../Helper/_helper.hpp"

class Timer
{

private:

    std::chrono::duration<double> _overall_time{};
    std::chrono::duration<double> _force_calculations{};
    std::chrono::duration<double> _making_neighbour_list{};
    std::chrono::duration<double> _updating_neighbour_list{};

public:
    Timer() = default;

    // ---- update functions ----
    template <typename Rep, typename Period>
    void update_overall_time(const std::chrono::duration<Rep, Period>& time_elapsed)
    {
        _overall_time += std::chrono::duration_cast<std::chrono::duration<double>>(time_elapsed);
    }

    template <typename Rep, typename Period>
    void update_force_calculations(const std::chrono::duration<Rep, Period>& time_elapsed)
    {
        _force_calculations += std::chrono::duration_cast<std::chrono::duration<double>>(time_elapsed);
    }

    template <typename Rep, typename Period>
    void update_making_neighbour_list(const std::chrono::duration<Rep, Period>& time_elapsed)
    {
        _making_neighbour_list += std::chrono::duration_cast<std::chrono::duration<double>>(time_elapsed);
    }

    template <typename Rep, typename Period>
    void update_updating_neighbour_list(const std::chrono::duration<Rep, Period>& time_elapsed)
    {
        _updating_neighbour_list += std::chrono::duration_cast<std::chrono::duration<double>>(time_elapsed);
    }


    void reset() noexcept
    {
        _overall_time = std::chrono::duration<double>::zero();
        _force_calculations = std::chrono::duration<double>::zero();
        _making_neighbour_list = std::chrono::duration<double>::zero();
        _updating_neighbour_list = std::chrono::duration<double>::zero();
    }

    void print_times()
    {
        std::cout << "Force calcs:      " << _force_calculations.count() << "\n";
        std::cout << "Making nl:        " << _making_neighbour_list.count() << "\n";
        std::cout << "Update nl:        " << _updating_neighbour_list.count() << "\n";
        std::cout << "Overall time:     " << _overall_time.count() << "\n";
    }
};


SINGLETON(Timer);

#endif 

