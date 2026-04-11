#ifndef MACROS_HPP
#define MACROS_HPP


/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
/*********************************************************************************************************************************/



#define CLASS_SET_GET(Class, attribute)                                      \
    inline void set_##attribute(Class& attribute)                            \
    {                                                                        \
        _##attribute = std::move(attribute);                                 \
    }                                                                        \
                                                                             \
    Class& get_##attribute()                                                 \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    const Class& get_##attribute() const                                     \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    Class get_##attribute##_copy() const                                     \
    {                                                                        \
        return _##attribute;                                                 \
    }


#define DOUBLE_SET_GET(attribute)                                            \
    inline void set_##attribute(double attribute)                            \
    {                                                                        \
        _##attribute = attribute;                                            \
    }                                                                        \
                                                                             \
    double get_##attribute() const                                    \
    {                                                                        \
        return _##attribute;                                                 \
    }


#define SIZE_T_SET_GET(attribute)                                            \
    inline void set_##attribute(std::size_t attribute)                            \
    {                                                                        \
        _##attribute = attribute;                                            \
    }                                                                        \
                                                                             \
    std::size_t get_##attribute() const                                    \
    {                                                                        \
        return _##attribute;                                                 \
    }


#define STRING_SET_GET(attribute)                                            \
    inline void set_##attribute(std::string& attribute)                            \
    {                                                                        \
        _##attribute = std::move(attribute);                                 \
    }                                                                        \
                                                                             \
    std::string& get_##attribute()                                                 \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    const std::string& get_##attribute() const                                     \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    std::string get_##attribute##_copy() const                                     \
    {                                                                        \
        return _##attribute;                                                 \
    }

    
#define ARRAY9_SET_GET(attribute)                                            \
    inline void set_##attribute(const std::array<double, 9>& attribute)      \
    {                                                                        \
        _##attribute = attribute;                                            \
    }                                                                        \
                                                                             \
    inline const std::array<double, 9>& get_##attribute() const              \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    inline std::array<double, 9>& get_##attribute()                          \
    {                                                                        \
        return _##attribute;                                                 \
    }



#define SINGLETON(Class)  \
    class Class##Once  \
    {  \
    public:  \
        inline static Class& get() { static Class instance; return instance; }   \
        inline static const Class& get_ro() { static Class instance; return instance; }   \
    private: \
        Class##Once() {}; \
        ~Class##Once() = default; \
        Class##Once(const Class##Once&) = delete; \
        Class##Once& operator=(const Class##Once&) = delete; \
        Class##Once(Class##Once&&) = delete; \
        Class##Once& operator=(const Class##Once&&) = delete; \
    };



#define ITERATOR(Type, Attribute) \
    std::vector<Type>::iterator begin() { return Attribute.begin(); } \
    std::vector<Type>::iterator end()   { return Attribute.end(); } \
    std::vector<Type>::const_iterator begin() const { return Attribute.begin(); } \
    std::vector<Type>::const_iterator end()   const { return Attribute.end(); } \
    std::vector<Type>::const_iterator cbegin() const { return Attribute.cbegin(); } \
    std::vector<Type>::const_iterator cend()   const { return Attribute.cend(); }




#endif