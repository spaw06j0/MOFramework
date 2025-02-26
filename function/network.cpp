#include "network.h"
#include <algorithm>

Network::Network(std::vector<Layer*> layers) {
    this->layers = layers;
}

Network::~Network() {}

Matrix Network::forward(Matrix input_tensor)
{
    for (size_t i = 0; i < layers.size(); i++) {
        input_tensor = (*layers[i])(input_tensor);
    }
    return input_tensor;
}

std::vector<std::vector<Matrix>> Network::backward(Matrix Gradients)
{
    std::vector<std::vector<Matrix>> gradients;
    for (int i = layers.size() - 1; i >= 0; i--) {
        std::pair<Matrix, std::vector<Matrix>> return_data = layers[i]->backward(Gradients);
        Gradients = return_data.first;
        if (!layers[i]->getHasTrainableVar() && !return_data.second.empty()) {
            throw std::runtime_error("non-trainable layer should not have variable gradient\n");
        }
        gradients.push_back(return_data.second);
    }
    std::reverse(gradients.begin(), gradients.end());
    return gradients;
}

void Network::apply_gradients(std::vector<std::vector<Matrix>> gradients) {
    for (size_t i = 0; i < layers.size(); i++) {
        Layer *layer = layers[i];
        if (layer->getTrainableVar()) {
            layer->apply_gradient(gradients[i]);
        }
    }
}