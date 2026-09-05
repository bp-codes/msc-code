 
#include <iostream>
#include <algorithm>
#include <sycl/sycl.hpp>



/*
acpp -std=c++17 \
     -O3 \
     -v \
     --acpp-targets=cuda:sm_86 \
     list_devices.cpp \
     -o list_devices.x
*/


int main(int argc, char** argv) 
{

    for (const auto& p : sycl::platform::get_platforms()) 
    {
        std::cerr << "Platform: " << p.get_info<sycl::info::platform::name>() << "\n";
        for (const auto& d : p.get_devices()) 
        {
            std::cerr << "  Device: " << d.get_info<sycl::info::device::name>()
                << "  (cpu=" << d.is_cpu() << ", gpu=" << d.is_gpu() << ")\n";
        }
    }



}