// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>


int task(std::vector<int> numbers)
{
    int sum {};
    for(int i = 0; i <numbers.size(); i++)
    {
        sum = sum + numbers[i];
    }
    return sum;
}



int main(int argc, char** argv) 
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

    std::vector<int> numbers;
    numbers.reserve(N);

    int last {};
    for (int i = 0; i < N; ++i) 
    {
        int n = 8039 * (last + i + 550607) % 10000;
        numbers.push_back(n);
        last = n;
    }
    

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    int sum {};

    do 
    {
        sum = task(numbers);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);
   
    std::cout << "Serial," << iters << "," << sum << std::endl;
    
    return 0;
}
