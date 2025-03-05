#include "matrix.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <mkl.h>
#include <omp.h>
#include <pthread.h>
#include <cuda_runtime.h>

int Matrix::mulMode = Matrix::CUDA;

Matrix::Matrix() : row(0), col(0), data(nullptr) {}

Matrix::Matrix(size_t r, size_t c)
    : row(r), col(c),
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
// template<typename Type>
// Matrix::Matrix(Type* ptr, size_t r, size_t c)
//     : row(r), col(c),
//       data(nullptr)
// {
//     size_t element = row * col;
//     data = new double[element];
//     if (data != nullptr) {
//         for (size_t i = 0; i < element; i++) {
//             data[i] = (double)ptr[i];
//         }
//     }
//     else {
//         std::cout << "Out of memory" << std::endl;
//     }
// }

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

double Matrix::operator() (size_t r, size_t c) const
{
    if (r >= this->row || c >= this->col) {
        throw std::runtime_error("row or col out of bound");
    }
    return data[r * this->col + c];
}

double &Matrix::operator() (size_t r, size_t c)
{
    if (r >= this->row || c >= this->col) {
        throw std::runtime_error("row or col out of bound");
    }
    return data[r * this->col + c];
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
        size_t col_idx = i / col;
        size_t row_idx = i % col;
        temp(row_idx, col_idx) = data[i];
    }
    return temp;
}

Matrix Matrix::fillwith(size_t r, size_t c, double num)
{
    Matrix temp(r, c);
    for (size_t i = 0; i < r * c; i++) {
        temp.data[i] = num;
    }
    return temp;
}

Matrix Matrix::zeros(size_t r, size_t c)
{
    return fillwith(r, c, 0.0);
}

Matrix Matrix::ones(size_t r, size_t c)
{
    return fillwith(r, c, 1.0);
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
Matrix mat_multiply(const Matrix &mat1, const Matrix &mat2) {
    // std::cout << "Matrix::mulMode: " << Matrix::mulMode << std::endl;
    switch (Matrix::mulMode) {
        case 0:
            return multiply(mat1, mat2);
        case 1:
            return multiply_mkl(mat1, mat2);
        case 2:
            return multiply_tile(mat1, mat2, 16);
        case 3:
            return multiply_openmp(mat1, mat2);
        case 4:
            return multiply_thread(mat1, mat2, 16);
        case 5:
            return multiply_cuda(mat1, mat2);
        default:
            throw std::runtime_error("Invalid multiplication mode");
    }
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

Matrix multiply_mkl(const Matrix &mat1, const Matrix &mat2) {
    size_t row = mat1.getRow();
    size_t col = mat2.getCol();
    size_t mid = mat1.getCol();
    if (mid != mat2.getRow()) {
        throw std::runtime_error("matrix dimension not match");
    }
    Matrix temp(row, col);
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        row,
        col,
        mid,
        1.0,
        mat1.data,
        mid,
        mat2.data,
        col,
        0.0,
        temp.data,
        col);
    return temp;
}

Matrix multiply_tile(const Matrix &mat1, const Matrix &mat2, size_t tile_size) {
    const size_t row1 = mat1.getRow();
    const size_t col1 = mat1.getCol();
    const size_t col2 = mat2.getCol();

    if (col1 != mat2.getRow()) {
        throw std::runtime_error("matrix dimension not match");
    }

    Matrix temp(row1, col2);
    // Process each tile
    for (size_t i = 0; i < row1; i += tile_size) {
        size_t i_end = std::min(i + tile_size, row1);
        for (size_t j = 0; j < col2; j += tile_size) {
            size_t j_end = std::min(j + tile_size, col2);
            for (size_t k = 0; k < col1; k += tile_size) {
                size_t k_end = std::min(k + tile_size, col1);
                // Process elements within the current tile
                for (size_t ii = i; ii < i_end; ii++) {
                    for (size_t jj = j; jj < j_end; jj++) {
                        double sum = 0.0;
                        for (size_t kk = k; kk < k_end; kk++) {
                            sum += mat1(ii, kk) * mat2(kk, jj);
                        }
                        temp(ii, jj) += sum;
                    }
                }
            }
        }
    }
    return temp;
}

Matrix multiply_openmp(const Matrix &mat1, const Matrix &mat2) {
    size_t row = mat1.getRow();
    size_t col = mat2.getCol();
    size_t mid = mat1.getCol();
    if (mid != mat2.getRow()) {
        throw std::runtime_error("matrix dimension not match");
    }
    Matrix temp(row, col);
    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     int num_threads = omp_get_num_threads();
    // }
    #pragma omp parallel for schedule(dynamic)
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
struct threadArgs {
    int threadId;
    int numThreads;
    int numBlock;

    const Matrix *mat1;
    const Matrix *mat2;
    Matrix *temp;

    size_t ntrow1;
    size_t ntcol1;
    size_t ntcol2;

    size_t rowMod;
    size_t colMod;
    size_t col2Mod;

    size_t rowFlag;
    size_t colFlag;
    size_t col2Flag;

    size_t sz;
};

void *threadMatMul(void *args) {
    threadArgs *targs = (threadArgs *)args;
    size_t row = targs->mat1->getRow();
    size_t mid = targs->mat1->getCol();
    size_t col = targs->mat2->getCol();

    size_t each = row / targs->numThreads;

    size_t start = targs->threadId * each;
    size_t end = targs->threadId == targs->numThreads - 1 ? row : start + each;

    for (size_t i = start; i < end; i++) {
        for (size_t j = 0; j < col; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < mid; k++) {
                sum += targs->mat1->data[i * mid + k] * targs->mat2->data[k * col + j];
            }
            targs->temp->data[i * col + j] = sum;
        }
    }
    pthread_exit((void *)0);
}

Matrix multiply_thread(const Matrix &mat1, const Matrix &mat2, int numThreads) {
    size_t row = mat1.getRow();
    size_t col = mat2.getCol();
    size_t mid = mat1.getCol();
    if (mid != mat2.getRow()) {
        throw std::runtime_error("matrix dimension not match");
    }
    Matrix temp(row, col);
    constexpr const int MAX_THREADS = 16;
    pthread_t threads[MAX_THREADS];
    threadArgs args[MAX_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int i = 0; i < numThreads; i++) {
        args[i].mat1 = &mat1;
        args[i].mat2 = &mat2;
        args[i].temp = &temp;
        args[i].threadId = i;
        args[i].numThreads = numThreads;
    }
    for (int i = 0; i < numThreads; i++) pthread_create(&threads[i], NULL, &threadMatMul, (void *)&args[i]);
    for (int i = 0; i < numThreads; i++) pthread_join(threads[i], NULL);
    pthread_attr_destroy(&attr);
    return temp;
}

extern "C" void launchMatrixMultiply(const double* mat1, const double* mat2, double* result, 
                                    size_t row, size_t mid, size_t col);


Matrix multiply_cuda(const Matrix &mat1, const Matrix &mat2) {
    size_t row = mat1.getRow();
    size_t col = mat2.getCol();
    size_t mid = mat1.getCol();
    if (mid != mat2.getRow()) {
        throw std::runtime_error("matrix dimension not match");
    }

    Matrix temp(row, col);

    double *d_mat1, *d_mat2, *d_temp;
    cudaMalloc((void **)&d_mat1, row * mid * sizeof(double));
    cudaMalloc((void **)&d_mat2, mid * col * sizeof(double));
    cudaMalloc((void **)&d_temp, row * col * sizeof(double));

    cudaMemcpy(d_mat1, mat1.data, row * mid * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2.data, mid * col * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((row + block.x - 1) / block.x, (col + block.y - 1) / block.y);
    launchMatrixMultiply(d_mat1, d_mat2, d_temp, row, mid, col);

    cudaMemcpy(temp.data, d_temp, row * col * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_temp);
    return temp;
}
