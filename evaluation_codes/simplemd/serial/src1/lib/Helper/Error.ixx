

export module error;

import <stdexcept>;
import <string>;

// Internal helper (not exported)
namespace error::detail
{
    inline std::string make_message(
        const char* kind,
        const std::string& msg,
        const char* file,
        int line,
        const char* function)
    {
        return std::string("\n[") + kind + "] \n"
             + "Error message:  " + msg
             + "\n  File: " + file
             + "\n  Line: " + std::to_string(line)
             + "\n  Function: " + function;
    }
}

// Exported throwing helpers
export inline void throw_invalid_argument(
    const std::string& msg,
    const char* file,
    int line,
    const char* function)
{
    throw std::invalid_argument(
        error::detail::make_message(
            "INVALID_ARGUMENT", msg, file, line, function));
}

export inline void throw_runtime_error(
    const std::string& msg,
    const char* file,
    int line,
    const char* function)
{
    throw std::runtime_error(
        error::detail::make_message(
            "RUNTIME_ERROR", msg, file, line, function));
}
 
