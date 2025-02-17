#include "layer.h"

class Sigmoid: public Layer
{
public:
    Sigmoid(): Layer(false, false) {}

    Matrix forward(const Matrix &input)
    {
        return input.sigmoid();
    }

    std::pair<Matrix, std::vector<Matrix>> backward(Matrix &gradient)
    {
        Matrix derivative = input.sigmoid() * (input.sigmoid() * -1.0 + 1.0) * gradient;
        return std::pair<Matrix, std::vector<Matrix>>(derivative, {});
    }
    
};

class ReLU: public Layer
{
public:
    ReLU(): Layer(false, false) {}
    
    Matrix forward(const Matrix &input)
    {
        return input.relu();
    }

    std::pair<Matrix, std::vector<Matrix>> backward(Matrix &gradient)
    {
        Matrix derivative = gradient;
        for(size_t i = 0; i < derivative.row; i++) {
            for(size_t j = 0; j < derivative.col; j++) {
                if (input(i,j) < 0.0)
                    derivative(i,j) = 0.0;
            }
        }
        return std::pair<Matrix, std::vector<Matrix>>(derivative, {});
    }
};

