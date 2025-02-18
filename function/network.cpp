#include "network.h"
#include <algorithm>

Network::Network(std::vector<Layer*> layers) {
    this->layers = layers;
}

Network::~Network() {}

Matrix Network::forward(Matrix &input)
{
    Matrix output = input;
    for (auto layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

std::vector<std::vector<Matrix>>  Network::backward(Matrix &Gradient)
{
    std::vector<std::vector<Matrix>> gradients;
    for (int i = layers.size() - 1; i >= 0; i--) {
        std::pair<Matrix, std::vector<Matrix>> return_data = layers[i]->backward(Gradient);
        Gradient = return_data.first;
        if (!layers[i]->getHasTrainableVar() && return_data.second.size() != 0) {
            throw std::runtime_error("no variable layaer should not have variable gradient\n");
        }
        gradients.push_back(return_data.second);
    }
    std::reverse(gradients.begin(), gradients.end());
    return gradients;
}

void Network::apply_gradient(std::vector<std::vector<Matrix>> &gradients) {
    for (int i = 0; i < layers.size(); i++) {
        if (layers[i]->getTrainable()) {
            layers[i]->apply_gradient(gradients[i]);
        }
    }
}