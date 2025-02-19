#include"matrix.h"

#ifndef __LOSS__
#define __LOSS__

class BaseLoss
{
public:
    BaseLoss() {};
    ~BaseLoss() {};
    Matrix operator() (const Matrix &prediction, const Matrix &ground_truth);

    virtual Matrix forward(const Matrix &prediction, const Matrix &ground_truth);
    virtual Matrix backward();

protected:
    Matrix gradient;
    Matrix input;
};

class MSE: public BaseLoss
{
public:
    using BaseLoss::BaseLoss;
    MSE(): BaseLoss() {};
    ~MSE() {};
    Matrix forward(const Matrix &prediction, const Matrix &ground_truth);
    Matrix backward();
};

class CategoricalCrossentropy: public BaseLoss
{
public:
    using BaseLoss::BaseLoss;
    CategoricalCrossentropy(): BaseLoss() {};
    ~CategoricalCrossentropy() {};
    Matrix forward(const Matrix &prediction, const Matrix &ground_truth);
    Matrix backward();
};

#endif