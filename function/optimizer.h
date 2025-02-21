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

// class Adam
// {
// public:
//     Adam(double learning_rate, double beta1, double beta2, double epsilon):
//         learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}
//     void apply_gradient(Network &network, std::vector<std::vector<Matrix>> gradients);
    
// private:
//     std::vector<std::vector<Matrix>> process_gradient(Network &network, std::vector<std::vector<Matrix>> gradient);
//     double learning_rate;
//     double beta1;
//     double beta2;
//     double epsilon;
//     double weight_decay;
//     int t;
//     std::vector<std::vector<Matrix>> m;
//     std::vector<std::vector<Matrix>> v;
// };
#endif