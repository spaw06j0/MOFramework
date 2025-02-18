#include "layer.h"
#include "activation.h"
#include <random>
#include <cassert>

class TestLayer : public Layer {
public:
    TestLayer() : Layer(true, true) {}

    Matrix forward(const Matrix &input) override {
        Matrix output(input.getRow(), input.getCol());
        for(size_t i = 0; i < input.getRow(); i++) {
            for(size_t j = 0; j < input.getCol(); j++) {
                output(i,j) = input(i,j) * 2.0;
            }
        }
        return output;
    }

    std::pair<Matrix, std::vector<Matrix>> backward(Matrix &gradient) override {
        return Layer::backward(gradient);
    } 
};

void testing() {
    TestLayer layer;
    // Test Constructor
    assert(layer.getTrainable() == true);
    assert(layer.getHasTrainableVar() == true);

    // Test forward
    Matrix input(2, 3);
    input.fillwith(2, 3, 1.0);
    Matrix output = layer.forward(input);
    assert(output.row == input.row);
    assert(output.col == input.col);
    assert(output(0,0) == 2.0); 

    // Test backward
    Matrix gradient(2, 3);
    gradient.fillwith(2, 3, 1.0);
    auto [d_input, d_gradient] = layer.backward(gradient);
    assert(d_input.row == input.row);
    assert(d_input.col == input.col);
    assert(d_gradient.size() == 1);

    std::vector<Matrix> weights = layer.get_weight();
    assert(weights.size() == 0); 
    
    std::cout << "All Layer tests passed!" << std::endl;
}
int main()
{
    testing();
    return 0;
}