// forward, backward, apply_gradient
// in_channel, out_channel, use_bias -> (Matrix) weight, bias | (Matrix) weight_gradient, bias_gradient
#include "layer.h"
#include "matrix.h"

#ifndef __LINEAR__
#define __LINEAR__

class Linear : public Layer {
public:
    Linear(int in_channel, int out_channel, bool use_bias = false, bool trainable = true);
    ~Linear();

    Matrix forward(const Matrix &input);
    std::pair<Matrix, std::vector<Matrix>> backward(Matrix &gradient);
    void apply_gradient(std::vector<Matrix> gradients);
    void set_weight(std::vector<Matrix> weight_list);
    std::vector<Matrix> get_weight();
    void print_weight_stats();

private:
    size_t inChannel;
    size_t outChannel;
    bool useBias;
    // forward
    Matrix weight;
    Matrix bias;
    // backward
    Matrix weightGradient;
    Matrix biasGradient;
};
#endif
