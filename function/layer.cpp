#include "layer.h"

Layer::Layer() {}
Layer::Layer(bool trainable, bool hasTrainableVar) : trainableVar(trainable), hasTrainableVar(hasTrainableVar) {}
Layer::~Layer() {}

Matrix Layer::operator()(Matrix &input_tensor)
{
    this->input = input_tensor;
    return this->forward(input_tensor);
}

Matrix Layer::forward(const Matrix &input_tensor)
{
    this->input = input_tensor;
    return input_tensor;
}

std::pair<Matrix, std::vector<Matrix>> Layer::backward(Matrix &gradient)
{
    return std::pair<Matrix, std::vector<Matrix>>(
        gradient,
        {gradient});
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


