#include <iostream>
#include <vector>
#include <omp.h>

int main()
{
    std::vector<double> data(1000, 1.0);
    auto data_sum {0.0};

    #pragma omp parallel for
    for(auto i = std::size_t(0); i < 1000; ++i)
    {
        data_sum = data_sum + data[i];
    }
    
    std::cout << "Sum: " << data_sum << std::endl;
    return 0;
} 
