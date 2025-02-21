#include "layer.h"

#ifndef __NETWORK__
#define __NETWORK__

class Network {
public:
    Network(std::vector<Layer*> layers);
    ~Network();

    Matrix forward(Matrix input);
    std::vector<std::vector<Matrix>> backward(Matrix gradient);
    std::vector<Layer*>& get_layers() {return layers;}
    void apply_gradient(std::vector<std::vector<Matrix>> gradients);
    
private:
    std::vector<Layer*> layers;
    
};

#endif