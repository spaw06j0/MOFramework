// base class for layers (linear, relu, etc)
// forward, backward, apply_gradient
#include "matrix.h"
#include <vector>

#ifndef __LAYER__
#define __LAYER__

class Layer {
public:
    Layer();
    Layer(bool trainable, bool hasTrainableVar);
    ~Layer();
    Matrix operator()(Matrix &input_tensor);

    bool getTrainableVar() const {return trainableVar;}
    bool getHasTrainableVar() const {return hasTrainableVar;}

    virtual Matrix forward(const Matrix &input_tensor);
    virtual std::pair<Matrix, std::vector<Matrix>> backward(Matrix &input_tensor);
    virtual void apply_gradient(std::vector<Matrix> gradients);
    virtual void set_weight(std::vector<Matrix> weight_list);
    virtual std::vector<Matrix> get_weight();

protected:
    Matrix input;
    bool trainableVar;
    bool hasTrainableVar;
};

#endif
