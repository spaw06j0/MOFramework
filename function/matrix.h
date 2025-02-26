// memory, operator

#include <iostream>
#include <cmath>

#ifndef __MATRIX__
#define __MATRIX__

class Matrix {
public:
    Matrix();
    Matrix(size_t r, size_t c);

    template<typename Type>
    Matrix(Type* ptr, size_t r, size_t c);
    Matrix(const Matrix &target);
    ~Matrix();

    // operator
    double operator() (size_t r, size_t c) const;
    double &operator() (size_t r, size_t c);

    bool operator==(const Matrix &mat) const;
    void operator=(const Matrix &mat);

    Matrix operator+(const Matrix &mat) const;
    Matrix &operator+=(const Matrix &mat);
    Matrix operator+(double num) const;
    Matrix &operator+=(double num);
    friend Matrix operator+(double num, const Matrix &mat);

    Matrix operator-(const Matrix &mat) const;
    Matrix &operator-=(const Matrix &mat);
    Matrix operator-(double num) const;
    Matrix &operator-=(double num);
    friend Matrix operator-(double num, const Matrix &mat);

    Matrix operator*(const Matrix &mat) const;
    Matrix &operator*=(const Matrix &mat);
    Matrix operator*(double num) const;
    Matrix &operator*=(double num);
    friend Matrix operator*(double num, const Matrix &mat);

    Matrix operator/(const Matrix &mat) const;
    Matrix &operator/=(const Matrix &mat);
    Matrix operator/(double num) const;
    Matrix &operator/=(double num);
    friend Matrix operator/(double num, const Matrix &mat);
    
    Matrix power(double p) const;
    Matrix exp() const;
    Matrix log() const;
    Matrix sigmoid() const;
    Matrix relu() const;

    Matrix T() const;
    double *accessData() {return data;}
    size_t getRow() const {return row;}
    size_t getCol() const {return col;}
    double *getData() const {return data;}
    void printShape() const {
        std::cout << "row: " << row << " col: " << col << std::endl;
    }
    
    static Matrix fillwith(size_t r, size_t c, double num);
    static Matrix zeros(size_t r, size_t c);
    static Matrix ones(size_t r, size_t c);

    Matrix slice(size_t start_row, size_t end_row) const;

    // double sum() const {
    //     double total = 0.0;
    //     for (size_t i = 0; i < row * col; i++) {
    //         total += data[i];
    //     }
    //     return total;
    // }

public:
    size_t row;
    size_t col;
    double *data;
};

Matrix multiply(const Matrix &mat1, const Matrix &mat2);

#endif
