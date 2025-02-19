#include "linear.h"

Linear::Linear(int in_channel, int out_channel, bool useBias, bool trainable):
    Layer(trainable, true), inChannel(in_channel), outChannel(out_channel), useBias(useBias) 
{
    weight = Matrix(in_channel, out_channel);
    if (useBias) {
        bias = Matrix(1, out_channel);
    }
}

Linear::~Linear() {}
// z = W^T * x + b
Matrix Linear::forward(const Matrix &input) {
    Layer::forward(input);
    // std::cout << "input: " << input.getRow() << " " << input.getCol() << std::endl;
    // std::cout << "weight: " << this->weight.getRow() << " " << this->weight.getCol() << std::endl;
    if (input.getCol() != this->weight.getRow()) {
        throw std::runtime_error("Input matrix column size does not match weight matrix row size\n");
    }
    // out = weight * input + bias
    // TODO: multiply operation need to be optimized by accelerated operation (e.g. parallel programming)
    //  matrix multiply operation
    // size_t row = input.getRow();
    // size_t col = this->weight.getCol();
    // size_t content = input.getCol();
    // Matrix output(row, col);
    // for (size_t i = 0; i < row; i++) {
    //     for (size_t j = 0; j < col; j++) {
    //         double sum = 0.0;
    //         for (size_t k = 0; k < content; k++) {
    //             sum += input(i, k) * weight(k, j);
    //         }
    //         output(i, j) = sum;
    //     }
    // }
    Matrix output = multiply(input, this->weight);
    if (useBias) {
        output = output + this->bias;
    }
    return output;
}

std::pair<Matrix, std::vector<Matrix>> Linear::backward(Matrix &gradient) {
    // gradient is dL/dz from next layer
    
    // For weights: dL/dw = x^T * dL/dz
    // Since forward: z = xW, backward needs x^T
    // matrix multiply operation
    // weightGradient = input.T() * gradient;
    this->weightGradient = multiply(this->input.T(), gradient);
    // For input: dL/dx = dL/dz * W^T
    // Since forward: z = xW, backward needs W^T
    // Matrix dzdx = gradient * weight.T();
    Matrix dzdx = multiply(gradient, this->weight.T());
    // For bias: dL/db = sum(dL/dz) across batch dimension
    // Since forward: z = xW + b, backward sums the gradients
    if (useBias) {
        Matrix ones = Matrix(1, gradient.getRow());
        ones.fillwith(1, gradient.getRow(), 1.0);
        this->biasGradient = multiply(ones, gradient);
        return std::pair<Matrix, std::vector<Matrix>>(
            dzdx,
            {this->weightGradient, this->biasGradient}
        );
    }
    return std::pair<Matrix, std::vector<Matrix>>(
        dzdx,
        {this->weightGradient}
    );
}

void Linear::set_weight(std::vector<Matrix> weight_list) {
    Matrix &weight = weight_list[0];
    if (weight.getRow() != inChannel || weight.getCol() != outChannel) {
        throw std::runtime_error("Invalid weight matrix shape\n");
    }
    this->weight = weight;

    if (useBias) {
        Matrix &bias = weight_list[1];
        if (bias.getRow() != 1 || bias.getCol() != outChannel) {
            throw std::runtime_error("Invalid bias matrix shape\n");
        }
        this->bias = bias;
    }
}

void Linear::apply_gradient(std::vector<Matrix> gradients) {
    Matrix &w_grad = gradients[0];
    this->weight -= w_grad;
    if (useBias) {
        Matrix &b_grad = gradients[1];
        this->bias -= b_grad;
    }
}

std::vector<Matrix> Linear::get_weight()
{
    if (useBias) {
        return std::vector<Matrix>({weight, bias});
    }
    return std::vector<Matrix>({weight});

}
