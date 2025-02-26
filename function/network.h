#include "layer.h"

#ifndef __NETWORK__
#define __NETWORK__

class Network {
public:
    Network(std::vector<Layer*> layers);
    ~Network();

    Matrix forward(Matrix input_tensor);
    std::vector<std::vector<Matrix>> backward(Matrix Gradients);
    std::vector<Layer*>& get_layers() {return layers;}
    void apply_gradients(std::vector<std::vector<Matrix>> gradients);
    
private:
    std::vector<Layer*> layers;
    
};

#endif