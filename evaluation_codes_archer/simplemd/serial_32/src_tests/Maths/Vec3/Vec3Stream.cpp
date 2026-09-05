#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
#include <sstream>

// ---- Simple positive values ----
TEST(Vec3Ostream, OutputsSimpleValues)
{
    const Maths::Vec3 v(1.0, 2.0, 3.0);
    std::ostringstream oss;
    oss << v;
    const std::string found {oss.str()};
    const std::string expected {"[1, 2, 3]"};
    EXPECT_EQ(found, expected);
}

// ---- Mixed positive/negative/zero ----
TEST(Vec3Ostream, OutputsMixedValues)
{
    const Maths::Vec3 v(-1.5, 0.0, 2.25);
    std::ostringstream oss;
    oss << v;
    const std::string found {oss.str()};
    const std::string expected {"[-1.5, 0, 2.25]"};
    EXPECT_EQ(found, expected);
}

// ---- Large magnitude values ----
TEST(Vec3Ostream, OutputsLargeValues)
{
    const Maths::Vec3 v(1.0e12, -2.0e12, 3.0e12);
    std::ostringstream oss;
    oss << v;
    const std::string found {oss.str()};
    const std::string expected {"[1e+12, -2e+12, 3e+12]"};
    EXPECT_EQ(found, expected);
}
