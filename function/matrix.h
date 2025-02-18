// memory, operator

#include <iostream>

#ifndef __MATRIX__
#define __MATRIX__

class Matrix {
public:
    Matrix();
    Matrix(size_t row, size_t col);
    template<typename Type>
    Matrix(Type* ptr, size_t row, size_t col);
    Matrix(const Matrix &target);
    ~Matrix();

    // operator
    double operator() (size_t row, size_t col) const;
    double &operator() (size_t row, size_t col);

    bool operator==(const Matrix &mat) const;
    void operator=(const Matrix &mat);

    Matrix operator+(const Matrix &mat) const;
    Matrix &operator+=(const Matrix &mat);
    Matrix operator+(double num) const;
    Matrix &operator+=(double num);

    Matrix operator-(const Matrix &mat) const;
    Matrix &operator-=(const Matrix &mat);
    Matrix operator-(double num) const;
    Matrix &operator-=(double num);

    Matrix operator*(const Matrix &mat) const;
    Matrix &operator*=(const Matrix &mat);
    Matrix operator*(double num) const;
    Matrix &operator*=(double num);

    Matrix operator/(const Matrix &mat) const;
    Matrix &operator/=(const Matrix &mat);
    Matrix operator/(double num) const;
    Matrix &operator/=(double num);
    
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

    void fillwith(size_t row, size_t col, double num);
    void zeros(size_t row, size_t col);
    void ones(size_t row, size_t col);

    Matrix slice(size_t start_row, size_t end_row) const;
public:
    size_t row;
    size_t col;
    double *data;
};

#endif
