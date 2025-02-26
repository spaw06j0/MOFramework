#include <iostream>
#include <cassert>
#include "../function/linear.h"

void test_linear_initialization() {
    std::cout << "Testing Linear layer Initialization..." << std::endl;

    // Test 1: Basic initialization with no bias
    Linear layer1(3, 2, false);
    auto weights1 = layer1.get_weight();
    assert(weights1.size() == 1);
    assert(weights1[0].getRow() == 3);
    assert(weights1[0].getCol() == 2);

    // Test 2: Initialization with bias
    Linear layer2(3, 2, true);
    auto weights2 = layer2.get_weight();
    assert(weights2.size() == 2);
    assert(weights2[0].getRow() == 3);
    assert(weights2[0].getCol() == 2);
    assert(weights2[1].getRow() == 1);
    assert(weights2[1].getCol() == 2);

    std::cout << "Linear layer Initialization test passed!" << std::endl;
}

/**
 * Linear Layer Correctness Verification
 * 
 * Mathematical Properties to Verify:
 * 1. Forward pass: y = xW + b
 * 2. Backward pass gradients:
 *    - dL/dx = dL/dy * W^T
 *    - dL/dW = x^T * dL/dy
 *    - dL/db = sum(dL/dy, dim=0)
 */

void verify_linear_layer_correctness() {
    // 1. Verify Forward Pass
    {
        
        Linear layer(2, 3, true);  // 2 input features, 3 output features
        
        // Set weights manually for verification
        Matrix weight = Matrix::fillwith(2, 3, 1.0);
        Matrix bias = Matrix::fillwith(1, 3, 0.5);
        layer.set_weight({weight, bias});
        
        // Input: batch_size=2, features=2
        Matrix input = Matrix::fillwith(2, 2, 2.0);
        
        Matrix output = layer.forward(input);
        
        // Expected: input * weight + bias
        // Each input element = 2.0
        // Each weight element = 1.0
        // Each bias element = 0.5
        // Expected output = (2.0 * 1.0 + 2.0 * 1.0) + 0.5 = 4.5
        assert(output.row == 2 && output.col == 3);
        for(size_t i = 0; i < output.row; i++) {
            for(size_t j = 0; j < output.col; j++) {
                assert(std::abs(output(i,j) - 4.5) < 1e-10);
            }
        }
    }
    std::cout << "Forward pass test passed!" << std::endl;
    // 2. Verify Backward Pass
    {
        Linear layer(2, 3, true);
        Matrix weight = Matrix::fillwith(2, 3, 1.0);
        Matrix bias = Matrix::fillwith(1, 3, 0.5);
        layer.set_weight({weight, bias});
        
        // Forward pass with known input
        Matrix input = Matrix::fillwith(4, 2, 1.0);
        layer.forward(input);
        
        // Backward pass with known gradient
        Matrix dL_dy = Matrix::fillwith(4, 3, 1.0);
        
        auto [dL_dx, param_grads] = layer.backward(dL_dy);
        
        // Verify gradient dimensions
        assert(dL_dx.row == 4 && dL_dx.col == 2);  // Same as input
        assert(param_grads[0].row == 2 && param_grads[0].col == 3);  // Same as weight
        assert(param_grads[1].row == 1 && param_grads[1].col == 3);  // Same as bias
        
        // Verify dL/dx = dL/dy * W^T
        Matrix expected_dL_dx = multiply(dL_dy, weight.T());
        assert(dL_dx == expected_dL_dx);
        
        // Verify dL/dW = x^T * dL/dy
        Matrix expected_dL_dW = multiply(input.T(), dL_dy);
        assert(param_grads[0] == expected_dL_dW);
        
        // Verify dL/db = sum(dL/dy, dim=0)
        Matrix ones = Matrix::fillwith(1, dL_dy.getRow(), 1.0);
        Matrix expected_dL_db = multiply(ones, dL_dy);
        assert(param_grads[1] == expected_dL_db);
    }
    std::cout << "Backward pass test passed!" << std::endl;
    // 3. Verify Weight Updates
    {
        Linear layer(2, 3, true);
        Matrix initial_weight = layer.get_weight()[0];
        
        // Create gradient
        Matrix weight_grad = Matrix::fillwith(2, 3, 0.1);
        Matrix bias_grad = Matrix::fillwith(1, 3, 0.1);
        
        // Apply gradient
        layer.apply_gradient({weight_grad, bias_grad});
        
        Matrix updated_weight = layer.get_weight()[0];
        
        // Verify weight update: w = w - grad
        for(size_t i = 0; i < initial_weight.row; i++) {
            for(size_t j = 0; j < initial_weight.col; j++) {
                assert(std::abs((initial_weight(i,j) - weight_grad(i,j)) - 
                              updated_weight(i,j)) < 1e-10);
            }
        }
    }
    std::cout << "Weight update test passed!" << std::endl;
}

void test_linear_layer() {
    // 1. Create a simple linear layer
    int in_features = 3;
    int out_features = 2;
    Linear layer(in_features, out_features, true); // with bias

    // 2. Set predetermined weights and bias for easy verification
    Matrix weight(in_features, out_features);
    weight(0,0) = 1.0; weight(0,1) = 2.0;
    weight(1,0) = 0.5; weight(1,1) = -1.0;
    weight(2,0) = -1.0; weight(2,1) = 1.0;

    Matrix bias(1, out_features);
    bias(0,0) = 0.5; bias(0,1) = -0.5;

    layer.set_weight({weight, bias});

    // 3. Create input data
    Matrix input(2, in_features); // batch size of 2
    input(0,0) = 1.0; input(0,1) = 2.0; input(0,2) = 3.0;
    input(1,0) = 0.5; input(1,1) = -1.0; input(1,2) = 1.0;

    // 4. Test forward pass
    Matrix output = layer.forward(input);
    
    // Expected output calculation:
    // For first sample: [1,2,3] * [1,2;0.5,-1;-1,1] + [0.5,-0.5] = [0.5,1.5]
    // For second sample: [0.5,-1,1] * [1,2;0.5,-1;-1,1] + [0.5,-0.5] = [0.25,0.5]
    
    std::cout << "Forward pass test:" << std::endl;
    std::cout << "Output shape: " << output.getRow() << "x" << output.getCol() << std::endl;
    std::cout << "Output values:" << std::endl;
    for(size_t i = 0; i < output.getRow(); i++) {
        for(size_t j = 0; j < output.getCol(); j++) {
            std::cout << output(i,j) << " ";
        }
        std::cout << std::endl;
    }

    // 5. Test backward pass
    Matrix gradient(2, out_features); // Simulate gradient from next layer
    gradient(0,0) = 1.0; gradient(0,1) = -1.0;
    gradient(1,0) = 0.5; gradient(1,1) = 0.5;

    auto [dx, param_grads] = layer.backward(gradient);

    std::cout << "\nBackward pass test:" << std::endl;
    std::cout << "Input gradient shape: " << dx.getRow() << "x" << dx.getCol() << std::endl;
    std::cout << "Weight gradient shape: " << param_grads[0].getRow() << "x" << param_grads[0].getCol() << std::endl;
    if (param_grads.size() > 1) {
        std::cout << "Bias gradient shape: " << param_grads[1].getRow() << "x" << param_grads[1].getCol() << std::endl;
    }

    // 6. Verify gradients numerically using finite differences
    const double epsilon = 1e-7;
    // Test one weight element
    Matrix original_weight = weight;
    weight(0,0) += epsilon;
    layer.set_weight({weight, bias});
    Matrix output_plus = layer.forward(input);
    
    weight(0,0) -= 2 * epsilon;
    layer.set_weight({weight, bias});
    Matrix output_minus = layer.forward(input);
    
    double numerical_grad = ((output_plus - output_minus) / (2 * epsilon)).sum();
    double analytical_grad = param_grads[0](0,0);
    
    std::cout << "\nGradient check:" << std::endl;
    std::cout << "Numerical gradient: " << numerical_grad << std::endl;
    std::cout << "Analytical gradient: " << analytical_grad << std::endl;
    std::cout << "Difference: " << std::abs(numerical_grad - analytical_grad) << std::endl;
}

int main() {
    try {
        test_linear_initialization();
        verify_linear_layer_correctness();
        test_linear_layer();
        std::cout << "All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
    }
    return 0;
}