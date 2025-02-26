#include "linear.h"
#include "matrix.h"
#include <random>

Linear::Linear(int in_channel, int out_channel, bool useBias, bool trainable):
    Layer(trainable, true), inChannel(in_channel), outChannel(out_channel), useBias(useBias) 
{
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = std::sqrt(2.0 / (in_channel + out_channel));
    std::uniform_real_distribution<> dis(-limit, limit);
    
    this->weight = Matrix(in_channel, out_channel);
    for(size_t i = 0; i < this->weight.getRow(); i++) {
        for(size_t j = 0; j < this->weight.getCol(); j++) {
            this->weight(i,j) = dis(gen);
        }
    }
    // random_bias=np.random.standard_normal((1, out_feat))*0.01 + 1 / out_feat
    if (useBias) {
        this->bias = Matrix(1, out_channel) + 1 / out_channel;
    }
}

Linear::~Linear() {}
// z = W^T * x + b
Matrix Linear::forward(const Matrix &input_tensor) {
    if (input_tensor.getCol() != this->weight.getRow()) {
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
    Matrix output = multiply(input_tensor, this->weight);
    if (this->useBias) {
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
        Matrix ones = Matrix::fillwith(1, gradient.getRow(), 1.0);
        this->biasGradient = multiply(ones, gradient);
        return std::pair<Matrix, std::vector<Matrix>>(
            dzdx,
            {this->weightGradient, this->biasGradient}
        );
    }
    return std::pair<Matrix, std::vector<Matrix>>(
        dzdx,
        {this->weightGradient});
}

void Linear::set_weight(std::vector<Matrix> weight_list) {
    Matrix &_weight = weight_list[0];
    if (_weight.getRow() != inChannel || _weight.getCol() != outChannel) {
        throw std::runtime_error("Linear::set_weight: Invalid weight matrix shape\n");
    }
    this->weight = _weight;

    if (useBias) {
        Matrix &_bias = weight_list[1];
        if (_bias.getRow() != 1 || _bias.getCol() != outChannel) {
            throw std::runtime_error("Linear::set_weight: Invalid bias matrix shape\n");
        }
        this->bias = _bias;
    }
}

void Linear::apply_gradient(std::vector<Matrix> gradients) {
    Matrix &w_grad = gradients[0];
    this->weight -= w_grad;
    if (this->useBias) {
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

void Linear::print_weight_stats() {
    double sum = 0.0;
    double max_val = -1e9;
    double min_val = 1e9;
    
    for(size_t i = 0; i < weight.getRow(); i++) {
        for(size_t j = 0; j < weight.getCol(); j++) {
            double val = weight(i,j);
            sum += val;
            max_val = std::max(max_val, val);
            min_val = std::min(min_val, val);
        }
    }
    
    std::cout << "Weight stats - Mean: " << sum/(weight.getRow()*weight.getCol()) 
              << " Max: " << max_val << " Min: " << min_val << std::endl;
}

std::pair<size_t, size_t> Linear::getChannel() {
    std::cout << "inChannel: " << inChannel << " outChannel: " << outChannel << std::endl;
    return std::make_pair(inChannel, outChannel);
}