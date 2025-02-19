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

void test_linear_forward() {
    std::cout << "Testing Linear layer Forward..." << std::endl;

    Linear layer(2, 3, true);
    Matrix weight(2, 3);
    weight.fillwith(2, 3, 1.0);
    Matrix bias(1, 3);
    bias.fillwith(1, 3, 0.5);

    Matrix input(4, 2);
    input.fillwith(4, 2, 1.0);
    layer.set_weight({weight, bias});
    Matrix output = layer.forward(input);
    assert(output.getRow() == 4);
    assert(output.getCol() == 3);
    // input is 4x2 matrix (batch size = 4, input size = 2) : value = 1.0
    // weight is 2x3 matrix (input size = 2, output size = 3) : value = 1.0
    // bias is 1x3 matrix (output size = 3) : value = 0.5
    // output should be 4x3 matrix : value = 2.0 + 0.5 = 2.5
    assert(output(0, 0) == 2.5);
    assert(output(0, 1) == 2.5);
    assert(output(0, 2) == 2.5);
    assert(output(1, 0) == 2.5);
    assert(output(1, 1) == 2.5);
    assert(output(1, 2) == 2.5);
    assert(output(2, 0) == 2.5);
    assert(output(2, 1) == 2.5);
    assert(output(2, 2) == 2.5);
    assert(output(3, 0) == 2.5);
    assert(output(3, 1) == 2.5);
    assert(output(3, 2) == 2.5);

    std::cout << "Linear layer Forward test passed!" << std::endl;
}

void test_linear_backward() {
    std::cout << "Testing Linear layer backward pass..." << std::endl;
    
    Linear layer(2, 3, true);
    
    // Set weights and perform forward pass first
    Matrix weight(2, 3);
    weight.fillwith(2, 3, 1.0);
    Matrix bias(1, 3);
    bias.fillwith(1, 3, 0.5);
    layer.set_weight({weight, bias});
    
    Matrix input(4, 2);
    input.fillwith(4, 2, 1.0);
    layer.forward(input);
    
    // Create gradient
    Matrix gradient(4, 3);
    gradient.fillwith(4, 3, 1.0);
    // std::cout << "gradient in test: " << gradient.getRow() << " " << gradient.getCol() << std::endl;
    auto [dx, param_grads] = layer.backward(gradient);
    
    // Check gradient shapes
    assert(dx.getRow() == 4 && dx.getCol() == 2);
    assert(param_grads[0].getRow() == 2 && param_grads[0].getCol() == 3);  // weight gradient
    assert(param_grads[1].getRow() == 1 && param_grads[1].getCol() == 3);  // bias gradient
    
    std::cout << "Linear layer backward pass test passed!" << std::endl;
}

void test_linear_apply_gradient() {
    std::cout << "Testing Linear layer Apply Gradient..." << std::endl;

    Linear layer(2, 3, true);
    Matrix weight(2, 3);
    weight.fillwith(2, 3, 1.0);
    Matrix bias(1, 3);
    bias.fillwith(1, 3, 1.0);

    layer.set_weight({weight, bias});

    Matrix gradient(2, 3);
    gradient.fillwith(2, 3, 0.1);
    Matrix bias_gradient(1, 3);
    bias_gradient.fillwith(1, 3, 0.1);

    layer.apply_gradient({gradient, bias_gradient});

    assert(layer.get_weight()[0].getRow() == 2);
    assert(layer.get_weight()[0].getCol() == 3);
    auto updated_weights = layer.get_weight();
    assert(updated_weights[0](0, 0) == 0.9);
    assert(updated_weights[0](0, 1) == 0.9);
    assert(updated_weights[0](0, 2) == 0.9);
    assert(updated_weights[1](0, 0) == 0.9);
    assert(updated_weights[1](0, 1) == 0.9);
    assert(updated_weights[1](0, 2) == 0.9);

    std::cout << "Linear layer Apply Gradient test passed!" << std::endl;
}

int main() {
    try {
        test_linear_initialization();
        test_linear_forward();
        test_linear_backward();
        test_linear_apply_gradient();
        std::cout << "All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
    }
    return 0;
}