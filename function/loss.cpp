#include"loss.h"

Matrix BaseLoss::operator()(const Matrix &prediction, const Matrix &ground_truth)
{
    this->input = prediction;
    return this->forward(prediction, ground_truth);
}

Matrix BaseLoss::forward(const Matrix &prediction, const Matrix &ground_truth)
{
    return ground_truth;
}

Matrix BaseLoss::backward()
{
    return gradient;
}

Matrix MSE::forward(const Matrix &prediction, const Matrix &ground_truth)
{
    Matrix result = (prediction-ground_truth).power(2.0);
    this->gradient = (prediction - ground_truth) * 2.0;
    return result;
}

Matrix MSE::backward()
{
    return this->gradient;
}

Matrix CategoricalCrossentropy::forward(const Matrix &prediction, const Matrix &ground_truth)
{
    Matrix mat_exp = prediction.exp();
    Matrix mat_exp_sum(prediction.getRow(), 1);
    for(size_t i = 0; i < prediction.getRow(); i++) {
        for(size_t j = 0; j < prediction.getCol(); j++) {
            mat_exp_sum(i,0) += mat_exp(i,j); 
        }
    }
    Matrix normalize = mat_exp / mat_exp_sum;
    this->gradient = normalize - ground_truth;
    return normalize.log() * ground_truth * -1.0;
}

Matrix CategoricalCrossentropy::backward()
{
    return this->gradient;
}

// class CategoricalCrossentropy: public BaseLoss {
// public:
//     CategoricalCrossentropy(): BaseLoss() {}
    
//     Matrix forward(const Matrix &prediction, const Matrix &ground_truth) {
//         // Store for backward pass
//         this->input = prediction;
//         this->truth = ground_truth;
        
//         // Compute cross entropy directly (assume input is already softmaxed)
//         Matrix loss(prediction.getRow(), 1);
//         for(size_t i = 0; i < prediction.getRow(); i++) {
//             double sample_loss = 0;
//             for(size_t j = 0; j < prediction.getCol(); j++) {
//                 if(ground_truth(i,j) > 0) {
//                     // Add small epsilon to avoid log(0)
//                     sample_loss -= ground_truth(i,j) * std::log(std::max(prediction(i,j), 1e-7));
//                 }
//             }
//             loss(i,0) = sample_loss;
//         }
//         return loss;
//     }

//     Matrix backward() {
//         // Gradient is (prediction - target)
//         return input - truth;
//     }
// private:
//     Matrix truth;
// };