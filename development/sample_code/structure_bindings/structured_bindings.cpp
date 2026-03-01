#include <iostream>
#include <tuple>
#include <utility>


std::pair<int, int> my_function(int i)
{
	return {i + 1, i - 1};
}

int main()
{
	std::tuple<int, float, double> tuple = {1, 2.0f, 3.0};
	std::pair<int, int> pair = {0, 0};
	
	auto [t1, t2, t3] = tuple;
	auto [p1, p2] = pair;
	
	auto [r1, r2] = my_function(1);
	
	  

	std::cout << t1 << "  " << t2 << "  " << t3 << std::endl;	
	std::cout << r1 << "  " << r2 << std::endl;

	return 0;
}