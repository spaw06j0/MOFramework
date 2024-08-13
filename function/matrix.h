// Declaration of Matrix
// constructor, destructor, operator, function
#include <cstddef>
#include <iostream>

#ifndef __MATRIX__
#define __MATRIX__

template<typename Type>
class Matrix
{
public:
    Matrix();
    Matrix(size_t rows, size_t col);

    ~Matrix();

    Matrix(size_t rows, size_t col);
    Matrix(Type* ptr, size_t rows, size_t col);
    Matrix(const Matrix& target);

    ~Matrix();

    // Define operator
    double operator() (size_t row, size_t col) const;
    double& operator() (size_t row, size_t col);

    Matrix operator+(const Matrix& mat) const;
    Matrix& operator+=(const Matrix& mat);
    Matrix operator+(double num) const;
    Matrix& operator+=(double num);
    friend Matrix operator+(double num, const Matrix& mat);

    Matrix operator-(const Matrix& mat) const;
    Matrix& operator-=(const Matrix& mat);
    Matrix operator-(double num) const;
    Matrix& operator-=(double num);
    friend Matrix operator-(double num, const Matrix& mat);

    Matrix operator*(const Matrix& mat) const;
    Matrix& operator*=(const Matrix& mat);
    Matrix operator*(double num) const;
    Matrix& operator*=(double num);
    friend Matrix operator*(double num, const Matrix& mat);

    Matrix operator/(const Matrix& mat) const;
    Matrix& operator/=(const Matrix& mat);
    Matrix operator/(double num) const;
    Matrix& operator/=(double num);
    friend Matrix operator/(double num, const Matrix& mat);

    bool operator==(const Matrix& target) const;
    void operator=(const Matrix& target);

    Matrix power(double p) const;
    Matrix exp() const;
    Matrix log() const;
    Matrix sigmoid() const;
    Matrix relu() const;

    // Define function
    Matrix T() const;
    double* data() { return m_data; }
    size_t row() const { return m_row; }
    size_t col() const { return m_col; }
    double* get_buffer() const {return m_data;}
    void print_shape(const char* mat_name="") const {std::cout << mat_name << " m_row: " << m_row << " m_col: " << m_col << std::endl; }

public:
    size_t m_row;
    size_t m_col;
    double * m_data;
};

#endif