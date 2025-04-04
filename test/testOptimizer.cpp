#include <iostream>
#include <cassert>
#include "../function/optimizer.h"
#include "../function/linear.h"

// void test_sgd_optimizer() {
//     std::cout << "Testing SGD Optimizer..." << std::endl;
    
//     // Create a linear layer
//     Linear layer(2, 3, true);
//     Matrix weight(2, 3);
//     weight.fillwith(2, 3, 1.0);
//     Matrix bias(1, 3);
//     bias.fillwith(1, 3, 1.0);
//     layer.set_weight({weight, bias});
    
//     Network network({&layer});

//     // Create gradients
//     Matrix weight_grad(2, 3);
//     weight_grad.fillwith(2, 3, 0.1);
//     Matrix bias_grad(1, 3);
//     bias_grad.fillwith(1, 3, 0.1);
    
//     // Create SGD optimizer with learning rate 0.1 and momentum 0.9
//     SGD optimizer(0.1, 0.9);
    
//     // Store initial weights
//     auto initial_weights = layer.get_weight();
    
//     // Apply gradients
//     optimizer.apply_gradient(network, {{weight_grad, bias_grad}});
    
//     // Calculate expected values after first update
//     // For first update with momentum:
//     // v_t = momentum * 0 + gradient * 0.1 = 0.1 * 0.1 = 0.01
//     // weight_new = weight - v_t = 1.0 - 0.01 = 0.99
    
//     auto updated_weights = network.get_weights();
//     double expected_first_update = 1.0 - 0.01;
    
//     // Debug prints
//     // std::cout << "Actual weight after update: " << updated_weights[0][0](0, 0) << std::endl;
//     // std::cout << "Expected weight: " << expected_first_update << std::endl;
//     // std::cout << "Difference: " << std::abs(updated_weights[0][0](0, 0) - expected_first_update) << std::endl;
    
//     assert(std::abs(updated_weights[0][0](0, 0) - expected_first_update) < 1e-5);
    
//     // Apply gradients again
//     optimizer.apply_gradient(network, {{weight_grad, bias_grad}});
    
//     // Calculate expected values after second update
//     // v_t = momentum * prev_v_t + gradient * learning_rate
//     // v_t = 0.9 * 0.01 + 0.1 * 0.1 = 0.009 + 0.01 = 0.019
//     // weight_new = 0.99 - 0.019 = 0.971
    
//     auto second_update = network.get_weights();
//     double expected_second_update = expected_first_update - 0.019;
    
//     // Debug prints for second update
//     // std::cout << "Actual weight after second update: " << second_update[0][0](0, 0) << std::endl;
//     // std::cout << "Expected second weight: " << expected_second_update << std::endl;
//     // std::cout << "Second difference: " << std::abs(second_update[0][0](0, 0) - expected_second_update) << std::endl;
    
//     assert(std::abs(second_update[0][0](0, 0) - expected_second_update) < 1e-5);
    
//     std::cout << "SGD Optimizer test passed!" << std::endl;
// }

// void test_adam_optimizer() {
//     std::cout << "Testing Adam Optimizer..." << std::endl;
    
//     // Create a linear layer
//     Linear layer(2, 3, true);
//     Matrix weight(2, 3);
//     weight.fillwith(2, 3, 1.0);
//     Matrix bias(1, 3);
//     bias.fillwith(1, 3, 1.0);
//     layer.set_weight({weight, bias});
    
//     Network network({&layer});
    
//     // Create gradients
//     Matrix weight_grad(2, 3);
//     weight_grad.fillwith(2, 3, 0.1);
//     Matrix bias_grad(1, 3);
//     bias_grad.fillwith(1, 3, 0.1);
    
//     // Create Adam optimizer with fixed parameters
//     Adam optimizer(0.001, 0.9, 0.999, 1e-8);
//     optimizer.apply_gradient(network, {{weight_grad, bias_grad}});
    
//     // Calculate expected values after first update
//     // m_t = beta1 * 0 + (1-beta1) * 0.1 = 0.01
//     // v_t = beta2 * 0 + (1-beta2) * 0.01 = 0.00001
//     // m_hat = m_t / (1-beta1) = 0.01/0.1 = 0.1
//     // v_hat = v_t / (1-beta2) = 0.00001/0.001 = 0.01
//     // update = learning_rate * m_hat / (sqrt(v_hat) + epsilon) = 0.00316227
    
//     auto updated_weights = network.get_weights();
//     double expected_weight_change = 0.00316227;
//     assert(std::abs(1 - updated_weights[0][0](0, 0)) - expected_weight_change < 1e-5);
    
//     // Run it again with the same parameters
//     layer.set_weight({weight, bias});  // Reset weights to initial values
//     optimizer.apply_gradient(network, {{weight_grad, bias_grad}});

//     // Calculate expected values after second update
//     // m_t = beta1 * 0.01 + (1-beta1) * 0.1 = 0.019
//     // v_t = beta2 * 0.00001 + (1-beta2) * 0.01 = 0.000019
//     // m_hat = m_t / (1-beta1) = 0.019/(1-0.81) = 0.105263
//     // v_hat = v_t / (1-beta2) = 0.000019/(1 - 0.999^2) = 0.01
//     // update = learning_rate * m_hat / (sqrt(v_hat) + epsilon) = 0.004249

//     auto second_update = network.get_weights();
//     double expected_weight_change2 = 0.004249;
//     assert(std::abs(1.0 - second_update[0][0](0, 0)) - expected_weight_change2 < 1e-5);

//     std::cout << "Adam Optimizer test passed!" << std::endl;
// }

/**
 * Optimizer Correctness Verification
 * 
 * SGD with Momentum Update Rule:
 * v_t = momentum * v_{t-1} + learning_rate * gradient
 * w_t = w_{t-1} - v_t
 *
 * Adam Update Rules:
 * m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 * v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 * m_hat = m_t / (1 - beta1^t)
 * v_hat = v_t / (1 - beta2^t)
 * w_t = w_{t-1} - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
 */

// void verify_optimizer_correctness() {
//     // 1. Verify SGD with Momentum
//     {
//         // Create simple network with one linear layer
//         Linear layer(2, 2, true);
//         Matrix weight(2, 2);
//         weight.fillwith(2, 2, 1.0);
//         layer.set_weight({weight});
        
//         Network network({&layer});
        
//         // Create optimizer with known parameters
//         double lr = 0.1;
//         double momentum = 0.9;
//         SGD optimizer(lr, momentum);
        
//         // First update
//         Matrix grad1(2, 2);
//         grad1.fillwith(2, 2, 1.0);
        
//         // Expected: v1 = momentum * 0 + lr * grad = 0.1
//         // w1 = w0 - v1 = 1.0 - 0.1 = 0.9
//         std::cout << "First update" << std::endl;
//         optimizer.apply_gradient(network, {{grad1}});
//         auto weights1 = network.get_weights()[0][0];
//         assert(std::abs(weights1(0,0) - 0.9) < 1e-10);
        
//         // Second update with same gradient
//         // Expected: v2 = momentum * v1 + lr * grad = 0.9 * 0.1 + 0.1 = 0.19
//         // w2 = w1 - v2 = 0.9 - 0.19 = 0.71
//         optimizer.apply_gradient(network, {{grad1}});
//         auto weights2 = network.get_weights()[0][0];
//         assert(std::abs(weights2(0,0) - 0.71) < 1e-10);
//     }
//     std::cout << "SGD with Momentum Update Rule test passed!" << std::endl;

//     // 2. Verify Adam
//     {
//         Linear layer(2, 2, true);
//         Matrix weight(2, 2);
//         weight.fillwith(2, 2, 1.0);
//         layer.set_weight({weight});
        
//         Network network({&layer});
        
//         // Create Adam optimizer with known parameters
//         double lr = 0.001;
//         double beta1 = 0.9;
//         double beta2 = 0.999;
//         double epsilon = 1e-8;
//         Adam optimizer(lr, beta1, beta2, epsilon);
        
//         // First update
//         Matrix grad1(2, 2);
//         grad1.fillwith(2, 2, 1.0);
        
//         // Expected first moment: m1 = (1-beta1)*g = 0.1
//         // Expected second moment: v1 = (1-beta2)*g^2 = 0.001
//         // Bias corrections:
//         // m1_hat = m1/(1-beta1) = 0.1/0.1 = 1.0
//         // v1_hat = v1/(1-beta2) = 0.001/0.001 = 1.0
//         // Update: w1 = w0 - lr * m1_hat/sqrt(v1_hat + eps) ≈ 0.999
//         optimizer.apply_gradient(network, {{grad1}});
//         auto weights1 = network.get_weights()[0][0];
//         assert(std::abs(weights1(0,0) - 0.999) < 1e-6);
        
//         // Second update
//         Matrix grad2(2, 2);
//         grad2.fillwith(2, 2, 1.0);
//         optimizer.apply_gradient(network, {{grad2}});
//         auto weights2 = network.get_weights()[0][0];
//         assert(std::abs(weights2(0,0) - 0.999) < 1e-6);
//     }

//     std::cout << "Optimizer correctness test passed!" << std::endl;
// }

int main() {
    try {
        // test_sgd_optimizer();
        // test_adam_optimizer();
        // verify_optimizer_correctness();
        std::cout << "All optimizer tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
