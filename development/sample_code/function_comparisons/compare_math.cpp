// g++ -std=c++20 -O2 compare_math.cpp -o compare_math.x && ./compare_math.x
// clang++ -std=c++20 -O2 compare_math.cpp -o compare_math.x && ./compare_math.x

#include <cmath>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
    void print_header(const std::string& title)
    {
        std::cout << "\n============================================================\n";
        std::cout << title << "\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << std::left
                  << std::setw(32) << "input(s)"
                  << std::setw(24) << "output1 (naive)"
                  << std::setw(24) << "output2 (better)"
                  << "abs diff\n";
        std::cout << "------------------------------------------------------------\n";
    }

    void print_row_1(const std::string& in,
                     double out1,
                     double out2)
    {
        const double diff = std::abs(out1 - out2);
        std::cout << std::left
                  << std::setw(30) << in << "  "
                  << std::setw(22) << out1 << "  "
                  << std::setw(22) << out2 << "  "
                  << diff << "\n";
    }

    void print_row_2(const std::string& in,
                     double out1,
                     double out2)
    {
        const double diff = std::abs(out1 - out2);
        std::cout << std::left
                  << std::setw(30) << in << "  "
                  << std::setw(22) << out1 << "  "
                  << std::setw(22) << out2 << "  "
                  << diff << "\n";
    }
}

int main()
{
    std::cout.setf(std::ios::scientific);
    std::cout << std::setprecision(17);

    // 1) log1p(x) vs log(1+x)
    {
        print_header("1) std::log(1+x)  vs  std::log1p(x)");

        const std::vector<double> xs{
            1e-16,
            1e-12,
            -1e-16
        };

        for (double x : xs)
        {
            const double out1 = std::log(1.0 + x);
            const double out2 = std::log1p(x);
            print_row_1("x=" + std::to_string(x), out1, out2);
        }
    }

    // 2) expm1(x) vs exp(x)-1
    {
        print_header("2) std::exp(x)-1  vs  std::expm1(x)");

        const std::vector<double> xs{
            1e-16,
            1e-10,
            -1e-16
        };

        for (double x : xs)
        {
            const double out1 = std::exp(x) - 1.0;
            const double out2 = std::expm1(x);
            print_row_1("x=" + std::to_string(x), out1, out2);
        }
    }

    // 3) hypot(x,y) vs sqrt(x*x + y*y)
    {
        print_header("3) std::sqrt(x*x+y*y)  vs  std::hypot(x,y)");

        struct Pair { double x; double y; };
        const std::vector<Pair> cases{
            { 1e308, 1e308 },   // naive overflows x*x, y*y
            { 1e-308, 1e-308 }, // naive underflows squares
            { 3.0, 4.0 }        // sanity check
        };

        for (const auto& c : cases)
        {
            const double out1 = std::sqrt(c.x * c.x + c.y * c.y);
            const double out2 = std::hypot(c.x, c.y);

            const std::string in =
                "x=" + std::to_string(c.x) + ", y=" + std::to_string(c.y);

            print_row_2(in, out1, out2);
        }
    }

    // 4) fma(a,b,c) vs a*b + c
    {
        print_header("4) a*b+c  vs  std::fma(a,b,c)");

        struct Triple { double a; double b; double c; };
        const std::vector<Triple> cases{
            // cancellation / rounding sensitivity
            { 1e308, 1e-308, -1.0 },                 // ideally ~0, tricky rounding paths
            { 9007199254740992.0, 1.0, 1.0 },        // 2^53, adding 1 shows rounding behaviour
            { 1e16, 1.0000000000000002, -1e16 }      // near-cancellation
        };

        for (const auto& t : cases)
        {
            const double out1 = (t.a * t.b) + t.c;
            const double out2 = std::fma(t.a, t.b, t.c);

            const std::string in =
                "a=" + std::to_string(t.a) + ", b=" + std::to_string(t.b) + ", c=" + std::to_string(t.c);

            print_row_2(in, out1, out2);
        }
    }

    // 5) pow(2,n) scaling vs ldexp/scalbn (exact power-of-two scaling)
    {
        print_header("5) x*pow(2,n)  vs  std::ldexp(x,n) (or std::scalbn)");

        struct Scale { double x; int n; };
        const std::vector<Scale> cases{
            { 1.0, 10 },
            { 0.1, 50 },
            { 1e-300, 200 }
        };

        for (const auto& s : cases)
        {
            const double out1 = s.x * std::pow(2.0, static_cast<double>(s.n));
            const double out2 = std::ldexp(s.x, s.n); // same idea as scalbn(x,n)

            const std::string in =
                "x=" + std::to_string(s.x) + ", n=" + std::to_string(s.n);

            print_row_2(in, out1, out2);
        }
    }

    // 6) atan(y/x) vs atan2(y,x)
    {
        print_header("6) std::atan(y/x)  vs  std::atan2(y,x)");

        struct Pair { double y; double x; };
        const std::vector<Pair> cases{
            {  1.0,  1.0 },  // Q1
            {  1.0, -1.0 },  // Q2 (atan(y/x) loses quadrant)
            { -1.0, -1.0 }   // Q3
        };

        for (const auto& p : cases)
        {
            const double out1 = std::atan(p.y / p.x);
            const double out2 = std::atan2(p.y, p.x);

            const std::string in =
                "y=" + std::to_string(p.y) + ", x=" + std::to_string(p.x);

            print_row_2(in, out1, out2);
        }
    }

    // 7) manual remainder (x - trunc(x/y)*y) vs fmod
    {
        print_header("7) x - trunc(x/y)*y  vs  std::fmod(x,y)");

        struct Pair { double x; double y; };
        const std::vector<Pair> cases{
            {  5.3,  2.0 },
            { -5.3,  2.0 },
            {  5.3, -2.0 }
        };

        for (const auto& p : cases)
        {
            const double q = std::trunc(p.x / p.y);
            const double out1 = p.x - q * p.y; // manual "fmod-like"
            const double out2 = std::fmod(p.x, p.y);

            const std::string in =
                "x=" + std::to_string(p.x) + ", y=" + std::to_string(p.y);

            print_row_2(in, out1, out2);
        }
    }

    // 8) manual remainder "nearest integer multiple" vs std::remainder
    //    (manual uses nearbyint which rounds to nearest, ties-to-even under default mode)
    {
        print_header("8) x - nearbyint(x/y)*y  vs  std::remainder(x,y)");

        struct Pair { double x; double y; };
        const std::vector<Pair> cases{
            {  5.3,  2.0 },
            {  5.0,  2.0 },  // tie-ish cases are interesting
            { -5.3,  2.0 }
        };

        for (const auto& p : cases)
        {
            const double n = std::nearbyint(p.x / p.y);
            const double out1 = p.x - n * p.y;           // "nearest multiple" style
            const double out2 = std::remainder(p.x, p.y); // defined by standard

            const std::string in =
                "x=" + std::to_string(p.x) + ", y=" + std::to_string(p.y);

            print_row_2(in, out1, out2);
        }
    }

    std::cout << "\nDone.\n";
    return 0;
} 
