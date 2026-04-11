#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>




int main(int argc, char** argv) 
{
    
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> c = {0.0, 0.0, 0.0, 0.0, 0.0};

    for(auto i = std::size_t(0); i < a.size(); ++i)
    {
        c[i] = a[i] + b[i];
    }

    for(const auto ci : c)
    {
        std::cout << ci << "  ";
    }
    std::cout << std::endl;
    
    return 0;
}
