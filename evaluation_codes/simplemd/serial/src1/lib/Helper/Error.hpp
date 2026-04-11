#ifndef ERROR_HPP
#define ERROR_HPP

import error;

#define THROW_INVALID_ARGUMENT(msg) \
    ::throw_invalid_argument((msg), __FILE__, __LINE__, __func__)

#define THROW_RUNTIME_ERROR(msg) \
    ::throw_runtime_error((msg), __FILE__, __LINE__, __func__)

#endif