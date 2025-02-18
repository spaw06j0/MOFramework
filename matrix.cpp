#include "matrix.h"
#include <iostream>
#include <cstring>
#include <cmath>

Matrix::Matrix() : row(0), col(0), data(nullptr) {}

Matrix::Matrix(size_t row, size_t col)
    : row(row), col(col),
      data(new double[row * col])
{   
    if (data != nullptr) {
        memset(data, 0, row * col * sizeof(double));
    }
    else {
        std::cout << "Out of memory" << std::endl;
    }
}
template<typename Type>
Matrix::Matrix(Type* ptr, size_t row, size_t col)
    : row(row), col(col),
      data(new double[row * col])
{
    for (size_t i = 0; i < row * col; i++) {
        data[i] = (double)ptr[i];
    }
}
Matrix::Matrix(const Matrix &target)
    : row(target.row), col(target.col),
      data(new double[target.row * target.col])
{
    if (data == nullptr) {
        std::cout << "Out of memory" << std::endl;
    }
    memcpy(data, target.data, sizeof(double) * row * col);
}

Matrix::~Matrix()
{
    delete[] data;
    row = col = 0;
    data = nullptr;
}

double Matrix::operator() (size_t row, size_t col) const
{
    if (row >= this->row || col >= this->col) {
        throw std::runtime_error("row or col out of bound");
    }
    return data[row * this->col + col];
}

double &Matrix::operator() (size_t row, size_t col)
{
    if (row >= this->row || col >= this->col) {
        throw std::runtime_error("row or col out of bound");
    }
    return data[row * this->col + col];
}

bool Matrix::operator==(const Matrix &target) const
{
    if (row != target.row || col != target.col) {
        return false;
    }
    else {
        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                if(data[i * col + j] != target.data[i * col + j]) {
                    return false;
                }
            }
        }
        return true;
    }
}

void Matrix::operator=(const Matrix &target)
{
    if (this == &target) return;
    
    delete[] data;
    row = target.row;
    col = target.col;
    data = new double[row * col];
    memcpy(data, target.data, sizeof(double) * row * col);
}

// Matrix Matrix::operator+(const Matrix &mat) const
// {
//     if (row != mat.row || col != mat.col) {
//         throw std::runtime_error("row or col not match");
//     }
//     Matrix temp(row, col);
//     for (size_t i = 0; i < row; i++) {
//         for (size_t j = 0; j < col; j++) {
//             temp(i, j) = data.get()[i * col + j] + mat.data.get()[i * col + j];
//         }
//     }
//     return temp;
// }

// Matrix &Matrix::operator+=(const Matrix &mat)
// {
//     if (row != mat.row || col != mat.col) {
//         throw std::runtime_error("row or col not match");
//     }
//     for (size_t i = 0; i < row; i++) {
//         for (size_t j = 0; j < col; j++) {
//             data.get()[i * col + j] += mat.data.get()[i * col + j];
//         }
//     }
//     return *this;
// }

// Matrix Matrix::operator+(double num) const
// {
//     Matrix temp(row, col);
//     for (size_t i = 0; i < row; i++) {
//         for (size_t j = 0; j < col; j++) {
//             temp(i, j) = data.get()[i * col + j] + num;
//         }
//     }
//     return temp;
// }

// Matrix &Matrix::operator+=(double num)
// {
//     for (size_t i = 0; i < row; i++) {
//         for (size_t j = 0; j < col; j++) {
//             data.get()[i * col + j] += num;
//         }
//     }
//     return *this;
// }

// mat + mat
#define MATRIX_OP_MATRIX(FUNCNAME, OP) \
Matrix Matrix::FUNCNAME(const Matrix &mat) const \
{ \
    if (row != mat.row || col != mat.col) { \
        throw std::runtime_error("row or col not match"); \
    } \
    Matrix temp(row, col); \
    for (size_t i = 0; i < row * col; i++) { \
        temp.data[i] = data[i] OP mat.data[i]; \
    } \
    return temp; \
} \

// mat += mat
#define MATRIX_ASSIGN_OP_MATRIX(FUNCNAME, OP) \
Matrix& Matrix::FUNCNAME(const Matrix &mat) \
{ \
    if (row != mat.row || col != mat.col) { \
        throw std::runtime_error("row or col not match"); \
    } \
    for (size_t i = 0; i < row * col; i++) { \
        data[i] OP mat.data[i]; \
    } \
    return *this; \
} \

// mat + num
#define MATRIX_OP_DOUBLE(FUNCNAME, OP) \
Matrix Matrix::FUNCNAME(double num) const \
{ \
    Matrix temp(row, col); \
    for (size_t i = 0; i < row * col; i++) { \
        temp.data[i] = num OP data[i]; \
    } \
    return temp; \
} \

// mat += num
#define MATRIX_ASSIGN_OP_DOUBLE(FUNCNAME, OP) \
Matrix& Matrix::FUNCNAME(double num) \
{ \
    for (size_t i = 0; i < row * col; i++) { \
        data[i] OP num; \
    } \
    return *this; \
} \

// +
MATRIX_OP_MATRIX(operator+, +)
MATRIX_ASSIGN_OP_MATRIX(operator+=, +=)
MATRIX_OP_DOUBLE(operator+, +)
MATRIX_ASSIGN_OP_DOUBLE(operator+=, +=)

// -
MATRIX_OP_MATRIX(operator-, -)
MATRIX_ASSIGN_OP_MATRIX(operator-=, -=)
MATRIX_OP_DOUBLE(operator-, -)
MATRIX_ASSIGN_OP_DOUBLE(operator-=, -=)

// *
MATRIX_OP_MATRIX(operator*, *)
MATRIX_ASSIGN_OP_MATRIX(operator*=, *=)
MATRIX_OP_DOUBLE(operator*, *)
MATRIX_ASSIGN_OP_DOUBLE(operator*=, *=)

// /
MATRIX_OP_MATRIX(operator/, /)
MATRIX_ASSIGN_OP_MATRIX(operator/=, /=)
MATRIX_OP_DOUBLE(operator/, /)
MATRIX_ASSIGN_OP_DOUBLE(operator/=, /=)

Matrix Matrix::power(double p) const
{
    Matrix temp(row, col);
    for (size_t i = 0; i < row * col; i++) {
        temp.data[i] = std::pow(data[i], p);
    }
    return temp;
}

Matrix Matrix::exp() const
{
    Matrix temp(row, col);
    for (size_t i = 0; i < row * col; i++) {
        temp.data[i] = std::exp(data[i]);
    }
    return temp;
}

Matrix Matrix::log() const
{
    Matrix temp(row, col);
    for (size_t i = 0; i < row * col; i++) {
        temp.data[i] = std::log(data[i]);
    }
    return temp;
}

Matrix Matrix::sigmoid() const
{
    Matrix temp(row, col);
    for (size_t i = 0; i < row * col; i++) {    
        temp.data[i] = 1.0 / (1.0 + std::exp(-data[i]));
    }
    return temp;
}

Matrix Matrix::relu() const
{
    Matrix temp(row, col);
    for (size_t i = 0; i < row * col; i++) {
        temp.data[i] = std::max(0.0, data[i]);
    }
    return temp;
}

Matrix Matrix::T() const
{
    Matrix temp(col, row);
    for (size_t i = 0; i < row * col; i++) {
        size_t col_idx = i / row;
        size_t row_idx = i % row;
        temp.data[col_idx * row + row_idx] = data[i];
    }
    return temp;
}

void Matrix::fillwith(size_t row, size_t col, double num)
{
    if (row != this->row || col != this->col) {
        throw std::runtime_error("row or col not match");
    }
    for (size_t i = 0; i < row * col; i++) {
        data[i] = num;
    }
}

void Matrix::zeros(size_t row, size_t col)
{
    fillwith(row, col, 0.0);
}

void Matrix::ones(size_t row, size_t col)
{
    fillwith(row, col, 1.0);
}



