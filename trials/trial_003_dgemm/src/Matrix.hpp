

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <vector>
#include <concepts>
#include <random>

template<typename T>
class Matrix
{

// Attributes (Private)
private:
    std::size_t _rows {};
    std::size_t _cols {};
    std::size_t _stride {};
    std::vector<T> _data {};        // Stored in ROW major (C/C++)


public:

    //       Constructors
    // ===============================
    Matrix() = default;

    Matrix(const std::size_t r, const std::size_t c)
        : _rows(r), _cols(c), _stride(c), _data(r * c)
    {}

    //       Accessors
    // ===============================

    // Rows, Cols, Stride
    [[nodiscard]] std::size_t rows() const noexcept { return _rows; }
    [[nodiscard]] std::size_t cols() const noexcept { return _cols; }
    [[nodiscard]] std::size_t stride() const noexcept { return _stride; }

    // Data
    [[nodiscard]] T* data() noexcept { return _data.data(); }
    [[nodiscard]] const T* data() const noexcept { return _data.data(); }
    [[nodiscard]] std::vector<T>& vector() noexcept { return _data; }
    [[nodiscard]] const std::vector<T>& vector() const noexcept { return _data; }



    // Create a Fortran column major 1D vector e.g. for cuBLAS
    [[nodiscard]] std::vector<T> vector_column_major() const noexcept 
    { 
        std::vector<T> data_column_major {};
        data_column_major.resize(_data.size());
        for (int i = 0; i < _rows; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                data_column_major[i + j * _rows] = _data[i * _cols + j];
            }
        }
        return data_column_major; 
    }



    // Load into existing matrix from 1D column major vector 
    void load_from_column_major(const std::vector<T>& data_column_major)
    {
        if (data_column_major.size() != _rows * _cols)
        {
            throw std::invalid_argument("Size mismatch in load_from_column_major");
        }

        for (std::size_t j = 0; j < _cols; ++j)
        {
            const std::size_t col_offset = j * _rows;
            for (std::size_t i = 0; i < _rows; ++i)
            {
                // column-major to row-major
                _data[i * _cols + j] = data_column_major[col_offset + i];
            }
        }
    }



    // Access elements
    [[nodiscard]] T& operator()(std::size_t i, std::size_t j)
    {
        return _data[i * _stride + j];
    }
    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const
    {
        return _data[i * _stride + j];
    }

    // Access elements (safe)
    T& at(std::size_t i, std::size_t j)
    {
        if (i >= _rows || j >= _cols)
            throw std::out_of_range("Matrix index out of range");

        return _data[i * _stride + j];
    }

    const T& at(std::size_t i, std::size_t j) const
    {
        if (i >= _rows || j >= _cols)
            throw std::out_of_range("Matrix index out of range");

        return _data[i * _stride + j];
    }

    void random_fill(std::mt19937_64& rng, std::uniform_real_distribution<double>& dist)
    {
        std::generate(_data.begin(), _data.end(), 
        [&]()
        {
            return static_cast<T>(dist(rng));
        });
    }



};




#endif