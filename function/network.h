#include "layer.h"

#ifndef NETWORK_H
#define NETWORK_H

class Network {
public:
    Network(std::vector<Layer*> layers);
    ~Network();
    Matrix forward(Matrix &input);
    std::vector<std::vector<Matrix>> backward(Matrix &gradient);
    std::vector<Layer*>& get_layers() {return layers; }
    void apply_gradient(std::vector<std::vector<Matrix>> &gradients);
    // get weights of all layers
    std::vector<std::vector<Matrix>> get_weights() {
        std::vector<std::vector<Matrix>> weights;
        for (auto layer : layers) {
            if (layer->getTrainable()) {
                weights.push_back(layer->get_weight());
            }
        }
        return weights;
    }
    
private:
    std::vector<Layer*> layers;
    
};

#endif