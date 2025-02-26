#include "optimizer.h"
#include <cmath>

// vt = momentum * vt-1 + learning_rate * gradient
void SGD::apply_gradient(Network &network, std::vector<std::vector<Matrix>> gradients)
{
    std::vector<std::vector<Matrix>> processed_grad = this->process_gradient(gradients);
    network.apply_gradients(processed_grad);
}

std::vector<std::vector<Matrix>> SGD::process_gradient(std::vector<std::vector<Matrix>> gradient)
{   
    std::vector<std::vector<Matrix>> new_gradient = gradient;
    for (size_t i = 0; i < new_gradient.size(); i++) {
        for (size_t j = 0; j < new_gradient[i].size(); j++) {
            Matrix &grad = new_gradient[i][j];
            grad = grad * learning_rate;
            if (previous_grad.size() != 0) {
                grad += previous_grad[i][j] * momentum;
            }
        }
    }
    previous_grad = new_gradient;
    return new_gradient;
}

// void Adam::apply_gradient(Network &network, std::vector<std::vector<Matrix>> gradients)
// {
//     std::vector<std::vector<Matrix>> processed_grad = this->process_gradient(network, gradients);
//     network.apply_gradient(processed_grad);
// }

// std::vector<std::vector<Matrix>> Adam::process_gradient(Network &network, std::vector<std::vector<Matrix>> gradient)
// {
//     t++;
//     if (m.empty()) {
//         m = gradient;
//         v = gradient;
//         for (auto& layer : m) {
//             for (auto& matrix : layer) {
//                 matrix.fillwith(matrix.getRow(), matrix.getCol(), 0);
//             }
//         }
//         for (auto& layer : v) {
//             for (auto& matrix : layer) {
//                 matrix.fillwith(matrix.getRow(), matrix.getCol(), 0);
//             }
//         }
//     }

//     std::vector<std::vector<Matrix>> new_gradient = gradient;
    
//     // Calculate bias corrections
//     float bc1 = 1.0f - std::pow(beta1, t);
//     float bc2 = 1.0f - std::pow(beta2, t);

//     for (size_t i = 0; i < new_gradient.size(); i++) {
//         for (size_t j = 0; j < new_gradient[i].size(); j++) {
//             Matrix& grad = new_gradient[i][j];
            
//             // Update biased first moment estimate
//             m[i][j] = m[i][j] * beta1 + grad * (1 - beta1);
            
//             // Update biased second raw moment estimate
//             Matrix grad_squared = grad.hadamard(grad);
//             v[i][j] = v[i][j] * beta2 + grad_squared * (1 - beta2);
            
//             // Compute bias-corrected moments
//             Matrix m_hat = m[i][j] * (1.0f / bc1);
//             Matrix v_hat = v[i][j] * (1.0f / bc2);
            
//             // Final update
//             Matrix sqrt_v = v_hat.sqrt();
//             sqrt_v = sqrt_v + epsilon;
//             grad = m_hat / sqrt_v;
//             grad = grad * learning_rate;
            
//             // Add weight decay term
//             Matrix weights = network.get_weights()[i][j];
//             grad = grad + weights * weight_decay * learning_rate;
//         }
//     }
    
//     return new_gradient;
// }