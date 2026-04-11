#include <iostream>
#include <omp.h>

int main()
{
    auto counter = std::size_t(0);

    #pragma omp parallel for
    for(auto i = std::size_t(0); i < 1000; ++i)
    {
        ++counter;
    }
    
    std::cout << "Counter value: " << counter << std::endl;
    return 0;
} 
