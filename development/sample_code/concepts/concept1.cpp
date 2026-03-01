// g++ --std=c++23 concept1.cpp -o concept1.x && ./concept1.x

#include <iostream>

template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
void print_numeric_type(T number)
{
	std::cout << number;
	
	// evaluated at compile time at template level
	if constexpr(std::is_same_v<T, int>)  
	{
		std::cout << ": int";
	}
	else if constexpr(std::is_same_v<T, float>)  
	{
		std::cout << ": float";
	}
	else
	{	
		std::cout << ": other";
	}
	std::cout << std::endl;
}

int main()
{
	return 0;
} 
