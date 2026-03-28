#ifndef ERROR_HPP
#define ERROR_HPP


#include <iostream>
#include <stdexcept>
#include <string>

// Helper for stringification
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Macro to throw std::invalid_argument with context
#define THROW_INVALID_ARGUMENT(msg) \
    throw std::invalid_argument( \
        std::string("\n[INVALID_ARGUMENT] \nError message:  ") + msg + \
        "\n  File: " + __FILE__ + \
        "\n  Line: " + TOSTRING(__LINE__) + \
        "\n  Function: " + __func__ \
    )


#define THROW_RUNTIME_ERROR(msg) \
    throw std::runtime_error( \
        std::string("\n[RUNTIME_ERROR] \nError message:  ") + msg + \
        "\n  File: " + __FILE__ + \
        "\n  Line: " + TOSTRING(__LINE__) + \
        "\n  Function: " + __func__ \
    )    



#endif