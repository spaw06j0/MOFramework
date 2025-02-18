#include "layer.h"

Layer::Layer() {}

Layer::~Layer() {}

Matrix Layer::forward(const Matrix &input)
{
    this->input = input;
    return input;
}

std::pair<Matrix, std::vector<Matrix>> Layer::backward(Matrix &gradient)
{
    return std::pair<Matrix, std::vector<Matrix>>(
        gradient,
        {gradient}
    );
}

void Layer::apply_gradient(std::vector<Matrix> gradients)
{
}

void Layer::set_weight(std::vector<Matrix> weight_list)
{
}

std::vector<Matrix> Layer::get_weight()
{
    return {};
}


