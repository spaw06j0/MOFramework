// implement CNN
#include <vector>
#include <cmath>
#include "matrix.h"

class ConvolutionalLayer {
private:
    int inputChannels, outputChannels, kernelSize, stride, padding;
    std::vector<Matrix> kernels;
    Matrix bias;

public:
    ConvolutionalLayer(int inChannels, int outChannels, int kSize, int s, int p)
        : inputChannels(inChannels), outputChannels(outChannels), kernelSize(kSize), stride(s), padding(p) {
        
        // Initialize kernels
        for (int i = 0; i < outputChannels; ++i) {
            kernels.push_back(Matrix(kernelSize, kernelSize));
            // Initialize weights (you may want to use a better initialization method)
            for (int j = 0; j < kernelSize * kernelSize; ++j) {
                kernels[i].m_buffer[j] = (double)rand() / RAND_MAX - 0.5;
            }
        }

        // Initialize bias
        bias = Matrix(1, outputChannels);
        for (int i = 0; i < outputChannels; ++i) {
            bias.m_buffer[i] = 0.0;
        }
    }

    Matrix forward(const Matrix& input) {
        int inputHeight = input.m_nrow;
        int inputWidth = input.m_ncol / inputChannels;
        int outputHeight = (inputHeight - kernelSize + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - kernelSize + 2 * padding) / stride + 1;

        Matrix output(outputHeight * outputChannels, outputWidth);

        // Implement convolution operation
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    double sum = 0.0;
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        for (int kh = 0; kh < kernelSize; ++kh) {
                            for (int kw = 0; kw < kernelSize; ++kw) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    sum += input.m_buffer[(ic * inputHeight + ih) * inputWidth + iw] *
                                           kernels[oc].m_buffer[kh * kernelSize + kw];
                                }
                            }
                        }
                    }
                    output.m_buffer[(oc * outputHeight + oh) * outputWidth + ow] = sum + bias.m_buffer[oc];
                }
            }
        }

        return output;
    }

    // Add methods for backward pass and parameter updates as needed
};

// You can add more CNN-related classes and functions here