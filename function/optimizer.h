#include "network.h"

#ifndef __OPTIMIZER__
#define __OPTIMIZER__

class SGD
{
public:
    SGD(double learning_rate, double momentum):
        learning_rate(learning_rate), momentum(momentum) {}
    void apply_gradient(Network &network, std::vector<std::vector<Matrix>> gradients);
    
private:
    std::vector<std::vector<Matrix>> process_gradient(std::vector<std::vector<Matrix>> gradient);
    std::vector<std::vector<Matrix>> previous_grad;
    double learning_rate;
    double momentum;
};

#endif