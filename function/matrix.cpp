#include "matrix.h"
#include <iostream>
#include <cstring>
#include <cmath>

Matrix::Matrix() : row(0), col(0), data(nullptr) {}

Matrix::Matrix(size_t row, size_t col)
    : row(row), col(col),
      data(nullptr)
{   
    size_t element = row * col;
    data = new double[element];
    if (data != nullptr) {
        memset(data, 0, element * sizeof(double));
    }
    else {
        std::cout << "Out of memory" << std::endl;
    }
}
template<typename Type>
Matrix::Matrix(Type* ptr, size_t row, size_t col)
    : row(row), col(col),
      data(nullptr)
{
    size_t element = row * col;
    data = new double[element];
    if (data != nullptr) {
        for (size_t i = 0; i < element; i++) {
            data[i] = (double)ptr[i];
        }
    }
    else {
        std::cout << "Out of memory" << std::endl;
    }
}

Matrix::Matrix(const Matrix &target)
{
    row = target.getRow();
    col = target.getCol();
    size_t element = row * col;
    data = new double[element];
    if (data != nullptr) {
        memcpy(data, target.data, sizeof(double) * element);
    }
    else {
        std::cout << "Out of memory" << std::endl;
    }
}

Matrix::~Matrix()
{
    delete[] data;
    row = col = 0;
    data = nullptr;
}

double Matrix::operator() (size_t row, size_t col) const
{
    if (row > this->row || col > this->col) {
        throw std::runtime_error("row or col out of bound");
    }
    return data[row * this->col + col];
}

double &Matrix::operator() (size_t row, size_t col)
{
    if (row > this->row || col > this->col) {
        throw std::runtime_error("row or col out of bound");
    }
    return data[row * this->col + col];
}

bool Matrix::operator==(const Matrix &target) const
{
    if (row != target.getRow() || col != target.getCol()) {
        return false;
    }
    else {
        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                if((*this)(i, j) != target(i, j)) {
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
    if (data != nullptr) {
        delete[] data;
    }
    row = target.getRow();
    col = target.getCol();
    size_t element = row * col;
    data = new double[element];
    memcpy(data, target.data, sizeof(double) * element);
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
    size_t row = std::max(mat.row, this->row); \
    size_t col = std::max(mat.col, this->col); \
    if (this->row != mat.row && !(mat.row == 1 || this->row == 1)) { \
        throw std::runtime_error("shape is not broadcastable in row"); \
    } \
    else if (this->col != mat.col && !(mat.col == 1 || this->col == 1)) { \
        throw std::runtime_error("shape is not broadcastable in col"); \
    } \
    Matrix temp(row, col); \
    if (mat.row == this->row && mat.col == this->col) { \
        for (size_t i = 0; i < this->row * this->col; i++) { \
            temp.data[i] = data[i] OP mat.data[i]; \
        } \
    } \
    else { \
        for (size_t i = 0; i < row; i++) { \
            for (size_t j = 0; j < col; j++) { \
                temp.data[i * col + j] = \
                    (*this)(i % this->row, j % this->col) OP \
                        mat(i % mat.row, j % mat.col); \
            } \
        } \
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
    for (size_t i = 0; i < this->row; i++) { \
        for (size_t j = 0; j < this->col; j++) { \
            data[i * this->col + j] OP mat(i, j); \
        } \
    } \
    return *this; \
} \

// mat + num
#define MATRIX_OP_DOUBLE(FUNCNAME, OP) \
Matrix Matrix::FUNCNAME(double num) const \
{ \
    Matrix temp(this->row, this->col); \
    for (size_t i = 0; i < this->row * this->col; i++) { \
        temp.data[i] = data[i] OP num; \
    } \
    return temp; \
} \

// num + mat
#define DOUBLE_OP_MATRIX(FUNCNAME, OP) \
Matrix FUNCNAME(double num, const Matrix &mat) \
{ \
    Matrix temp(mat); \
    for (size_t i = 0; i < mat.row * mat.col; i++) { \
        temp.data[i] = num OP mat.data[i]; \
    } \
    return temp; \
} \

// mat += num
#define MATRIX_ASSIGN_OP_DOUBLE(FUNCNAME, OP) \
Matrix& Matrix::FUNCNAME(double num) \
{ \
    for (size_t i = 0; i < this->row * this->col; i++) { \
        data[i] OP num; \
    } \
    return *this; \
} \

// +
MATRIX_OP_MATRIX(operator+, +)
MATRIX_ASSIGN_OP_MATRIX(operator+=, +=)
MATRIX_OP_DOUBLE(operator+, +)
DOUBLE_OP_MATRIX(operator+, +)
MATRIX_ASSIGN_OP_DOUBLE(operator+=, +=)

// -
MATRIX_OP_MATRIX(operator-, -)
MATRIX_ASSIGN_OP_MATRIX(operator-=, -=)
MATRIX_OP_DOUBLE(operator-, -)
DOUBLE_OP_MATRIX(operator-, -)
MATRIX_ASSIGN_OP_DOUBLE(operator-=, -=)

// *
MATRIX_OP_MATRIX(operator*, *)
MATRIX_ASSIGN_OP_MATRIX(operator*=, *=)
MATRIX_OP_DOUBLE(operator*, *)
DOUBLE_OP_MATRIX(operator*, *)
MATRIX_ASSIGN_OP_DOUBLE(operator*=, *=)

// /
MATRIX_OP_MATRIX(operator/, /)
MATRIX_ASSIGN_OP_MATRIX(operator/=, /=)
MATRIX_OP_DOUBLE(operator/, /)
DOUBLE_OP_MATRIX(operator/, /)
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
        auto x = std::exp(data[i]);
        if (std::isnan(x)) {
            temp.data[i] = 0.0;
        }
        else if (std::isinf(x)) {
            temp.data[i] = 10^5;
        }
        else {
            temp.data[i] = x;
        }
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

Matrix Matrix::fillwith(size_t row, size_t col, double num)
{
    Matrix temp(row, col);
    for (size_t i = 0; i < row * col; i++) {
        temp.data[i] = num;
    }
    return temp;
}

Matrix Matrix::zeros(size_t row, size_t col)
{
    return fillwith(row, col, 0.0);
}

Matrix Matrix::ones(size_t row, size_t col)
{
    return fillwith(row, col, 1.0);
}

Matrix Matrix::slice(size_t start_row, size_t end_row) const
{
    if (start_row < 0 || end_row > row || start_row >= end_row) {
        throw std::runtime_error("Invalid slice range");
    }
    Matrix temp(end_row - start_row, col);
    for (size_t i = 0; i < (end_row - start_row) * col; i++) {
        temp.data[i] = data[start_row * col + i];
    }
    return temp;
}

Matrix multiply(const Matrix &mat1, const Matrix &mat2) 
{
    size_t row = mat1.getRow();
    size_t col = mat2.getCol();
    size_t mid = mat1.getCol();
    if (mid != mat2.getRow()) {
        throw std::runtime_error("matrix dimension not match");
    }
    Matrix temp(row, col);
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < mid; k++) {
                sum += mat1(i, k) * mat2(k, j);
            }
            temp(i, j) = sum;
        }
    }
    return temp;
}