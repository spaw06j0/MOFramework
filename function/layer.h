// base class for layers (linear, relu, etc)
// forward, backward, apply_gradient
#include "matrix.h"
#include <vector>

#ifndef __LAYER__
#define __LAYER__

class Layer {
public:
    Layer();
    Layer(bool trainable, bool hasTrainableVar): trainable(trainable), hasTrainableVar(hasTrainableVar) {}
    virtual ~Layer();
    bool getTrainable() const {return trainable;}
    bool getHasTrainableVar() const {return hasTrainableVar;}

    virtual Matrix forward(const Matrix &input);
    virtual std::pair<Matrix, std::vector<Matrix>> backward(Matrix &gradient);
    virtual void apply_gradient(std::vector<Matrix> gradients);
    virtual void set_weight(std::vector<Matrix> weight_list);
    virtual std::vector<Matrix> get_weight();

protected:
    Matrix input;
    bool trainable;
    bool hasTrainableVar;
};

#endif
