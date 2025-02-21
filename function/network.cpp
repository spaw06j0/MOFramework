#include "network.h"

Network::Network(std::vector<Layer*> layers) {
    this->layers = layers;
}

Network::~Network() {}

Matrix Network::forward(Matrix input)
{
    for (auto layer : layers) {
        input = (*layer)(input);
    }
    return input;
}

std::vector<std::vector<Matrix>> Network::backward(Matrix Gradient)
{
    std::vector<std::vector<Matrix>> gradients(layers.size());
    for (int i = layers.size() - 1; i >= 0; i--) {
        std::pair<Matrix, std::vector<Matrix>> return_data = layers[i]->backward(Gradient);
        Gradient = return_data.first;
        if (!layers[i]->getHasTrainableVar() && !return_data.second.empty()) {
            throw std::runtime_error("non-trainable layer should not have variable gradient\n");
        }
        gradients[i] = return_data.second;
    }
    return gradients;
}

void Network::apply_gradient(std::vector<std::vector<Matrix>> gradients) {
    for (size_t i = 0; i < layers.size(); i++) {
        Layer *layer = layers[i];
        if (layer->getTrainable()) {
            layer->apply_gradient(gradients[i]);
        }
    }
}