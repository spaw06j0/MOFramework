#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h> 
#include <iostream>
#include <fstream>

#include "matrix.h"
#include "activation.h"
#include "layer.h"
#include "linear.h"
#include "network.h"
#include "loss.h"
#include "optimizer.h"

std::pair<Matrix, Matrix> load_mnist_data(const std::string& images_path, const std::string& labels_path, int num_samples) {
    // Open files
    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);
    
    if (!images_file.is_open() || !labels_file.is_open()) {
        throw std::runtime_error("Cannot open MNIST data files");
    }

    // Read headers
    uint32_t magic_number, n_images, n_rows, n_cols;
    images_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    images_file.read(reinterpret_cast<char*>(&n_images), sizeof(n_images));
    images_file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    images_file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

    // Convert from big-endian to host byte order
    magic_number = __builtin_bswap32(magic_number);
    n_images = __builtin_bswap32(n_images);
    n_rows = __builtin_bswap32(n_rows);
    n_cols = __builtin_bswap32(n_cols);

    // Skip label file header
    labels_file.seekg(8);

    // Prepare matrices
    Matrix images(num_samples, 784);  // 28*28 = 784
    Matrix labels(num_samples, 10);   // 10 classes for digits 0-9

    // Read data
    for (int i = 0; i < num_samples; i++) {
        // Read image
        unsigned char pixel;
        for (int j = 0; j < 784; j++) {
            images_file.read(reinterpret_cast<char*>(&pixel), 1);
            images(i, j) = static_cast<double>(pixel) / 255.0 - 0.5;  // Simple [0,1] normalization
        }

        // Read and one-hot encode label
        unsigned char label;
        labels_file.read(reinterpret_cast<char*>(&label), 1);
        for (int j = 0; j < 10; j++) {
            labels(i, j) = (j == label) ? 1.0 : 0.0;
        }
    }

    images_file.close();
    labels_file.close();

    return {images, labels}; // [batch, 784], [batch, 10]
}

float compute_accuracy(const Matrix& predictions, const Matrix& labels) {
    int correct = 0;
    int total = predictions.getRow();
    
    for(int i = 0; i < total; i++) {
        int pred_idx = 0;
        float max_val = predictions(i,0);
        for(int j = 1; j < 10; j++) {
            if(predictions(i,j) > max_val) {
                max_val = predictions(i,j);
                pred_idx = j;
            }
        }
        
        int true_idx = 0;
        for(int j = 0; j < 10; j++) {
            if(labels(i,j) == 1.0) {
                true_idx = j;
                break;
            }
        }
        
        if(pred_idx == true_idx) correct++;
    }
    
    return float(correct) / total;
}

namespace py = pybind11;
PYBIND11_MODULE(pynet, m) {
    m.doc() = "python binding for easy network";

    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        // matrix contain *data, row, col
        // conversion to numpy arrays
        .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.getData(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                2,
                {m.getRow(), m.getCol()},
                {sizeof(double) * m.getCol(), sizeof(double)}
            );
        })
        // Matrix(size_t r, size_t c);
        .def(py::init<size_t, size_t>())
        // Matrix(Type* ptr, size_t r, size_t c);
        .def(py::init([](py::buffer b)->Matrix {
            py::buffer_info info = b.request();
            size_t row = info.shape[0], col = info.shape[1];
            if (info.format == "L") // uint64
                return Matrix((uint64_t*)info.ptr, row, col);
            else if (info.format == "l") // int64
                return Matrix((int64_t*)info.ptr, row, col);
            else if (info.format == "I") // uint32
                return Matrix((uint32_t*)info.ptr, row, col);
            else if (info.format == "i") // int32
                return Matrix((int32_t*)info.ptr, row, col);
            else if (info.format == "H") // uint16
                return Matrix((uint16_t*)info.ptr, row, col);
            else if (info.format == "h") // int16
                return Matrix((int16_t*)info.ptr, row, col);
            else if (info.format == "B") // uint8
                return Matrix((uint8_t*)info.ptr, row, col);
            else if (info.format == "b") // int8
                return Matrix((int8_t*)info.ptr, row, col);
            else if (info.format == "f") // float32
                return Matrix((float*)info.ptr, row, col);
            else if (info.format == "d") // float64
                return Matrix((double*)info.ptr, row, col);
            else if (info.format == "g") // float128
                return Matrix((long double*)info.ptr, row, col);
            else 
                return Matrix();
            return Matrix();
        }))
        .def(py::init<const Matrix&>())
        // bool operator==(const Matrix &mat) const;
        .def("__eq__", &Matrix::operator==)
        
        //void operator=(const Matrix &mat);
        .def("__setitem__", [](Matrix &mat, std::pair<size_t, size_t> idx, double val) {return mat(idx.first, idx.second) = val;})
        .def("__getitem__", [](const Matrix &mat, std::pair<size_t, size_t> idx) {return mat(idx.first, idx.second);})

        .def("__add__", [](const Matrix &mat, int32_t num) {return mat + double(num);}, py::is_operator())
        .def("__sub__", [](const Matrix &mat, int32_t num) {return mat - double(num);}, py::is_operator())
        .def("__mul__", [](const Matrix &mat, int32_t num) {return mat * double(num);}, py::is_operator())
        .def("__truediv__", [](const Matrix &mat, int32_t num) {return mat / double(num);}, py::is_operator())

        // Matrix operator+(const Matrix &mat) const;
        .def(py::self + py::self)
        .def(py::self + double())
        .def(double() + py::self)
        .def(py::self += double())
        .def(py::self += py::self)

        .def(py::self - py::self)
        .def(py::self - double())
        .def(double() - py::self)
        .def(py::self -= double())
        .def(py::self -= py::self)

        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self *= double())
        .def(py::self *= py::self)

        .def(py::self / py::self)
        .def(py::self / double())
        .def(double() / py::self)
        .def(py::self /= double())
        .def(py::self /= py::self)

        .def("__add__", [](const Matrix &mat, int64_t num) {return mat + double(num);}, py::is_operator())
        .def("__sub__", [](const Matrix &mat, int64_t num) {return mat - double(num);}, py::is_operator())
        .def("__mul__", [](const Matrix &mat, int64_t num) {return mat * double(num);}, py::is_operator())
        .def("__truediv__", [](const Matrix &mat, int64_t num) {return mat / double(num);}, py::is_operator())
        
        .def("T", &Matrix::T)
        .def("getData", &Matrix::getData)
        .def("mean", &Matrix::mean)
        .def("power", &Matrix::power)
        .def("exp", &Matrix::exp)
        .def("log", &Matrix::log)
        .def("sigmoid", &Matrix::sigmoid)
        .def("relu", &Matrix::relu)

        .def("getRow", &Matrix::getRow)
        .def("getCol", &Matrix::getCol)
        .def_static("fillwith", &Matrix::fillwith)
        .def_static("zeros", &Matrix::zeros)
        .def_static("ones", &Matrix::ones)

        .def("slice", &Matrix::slice)
        .def("sum", &Matrix::sum)
        .def("mean", &Matrix::mean);

    m.def("multiply", &multiply, "Matrix multiplication");

    py::class_<Layer>(m, "Layer")
        .def(py::init<bool, bool>());

    py::class_<Linear, Layer>(m, "Linear")
        .def(py::init<int, int, bool, bool>())
        .def("forward", &Linear::forward)
        .def("__call__", &Linear::forward)
        .def("set_weight", &Linear::set_weight)
        .def("get_weight", &Linear::get_weight)
        .def_property_readonly("weight", &Linear::getWeight)
        .def_property_readonly("bias", &Linear::getBias);

    py::class_<Sigmoid, Layer>(m, "Sigmoid")
        .def(py::init<>())
        .def("__call__", &Sigmoid::forward);

    py::class_<ReLU, Layer>(m, "ReLU")
        .def(py::init<>())
        .def("__call__", &ReLU::forward);

    py::class_<Network>(m, "Network")
        .def(py::init<std::vector<Layer*>>(), py::keep_alive<1, 2>())
        .def("__call__", [](Network &net, const Matrix &mat1) {
            return net.forward(mat1);
        })
        .def("backward", &Network::backward)
        .def_property("layers", &Network::get_layers, nullptr);

    py::class_<BaseLoss>(m, "BaseLoss")
        .def(py::init<>())
        .def("__call__", &BaseLoss::operator())
        .def("forward", &BaseLoss::forward)
        .def("backward", &BaseLoss::backward);

    py::class_<MSE, BaseLoss>(m, "MSE")
        .def(py::init<>())
        .def("forward", &MSE::forward)
        .def("backward", &MSE::backward);

    py::class_<CategoricalCrossentropy, BaseLoss>(m, "CategoricalCrossentropy")
        .def(py::init<>())
        .def("forward", &CategoricalCrossentropy::forward)
        .def("backward", &CategoricalCrossentropy::backward);

    py::class_<SGD>(m, "SGD")
        .def(py::init<double, double>())
        .def("apply_gradient", &SGD::apply_gradient);
    
    m.def("load_mnist_data", &load_mnist_data,
        py::arg("images_path"),
        py::arg("labels_path"),
        py::arg("num_samples"));

    m.def("compute_accuracy", &compute_accuracy,
        py::arg("predictions"),
        py::arg("labels"));
}
    