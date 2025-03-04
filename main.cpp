#include "function/network.h"
#include "function/linear.h"
#include "function/activation.h"
#include "function/optimizer.h"
#include "function/loss.h"
#include "function/matrix.h"
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <cassert>

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

void check_data(const Matrix& images, const Matrix& labels, const std::string& name) {
    std::cout << "\nChecking " << name << " data:" << std::endl;
    
    // Check image values range
    double min_val = 1e9, max_val = -1e9;
    double sum = 0;
    for(size_t i = 0; i < images.getRow(); i++) {
        for(size_t j = 0; j < images.getCol(); j++) {
            double val = images(i,j);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
    }

    // Check label distribution
    std::vector<int> label_counts(10, 0);
    for(size_t i = 0; i < labels.getRow(); i++) {
        for(size_t j = 0; j < labels.getCol(); j++) {
            if(labels(i,j) > 0.5) {
                label_counts[j]++;
            }
        }
    }
    
    std::cout << "Label distribution:" << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << "Class " << i << ": " << label_counts[i] << " samples" << std::endl;
    }
}

// Helper function to compute accuracy
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

void test_compute_accuracy() {
    // Test case 1: Perfect predictions (100% accuracy)
    {
        // Create predictions with clear max values
        Matrix predictions(3, 10);
        predictions.fillwith(3, 10, 0.1); // Fill with low values
        predictions(0,2) = 0.9;  // Sample 1 predicts class 2
        predictions(1,5) = 0.9;  // Sample 2 predicts class 5
        predictions(2,8) = 0.9;  // Sample 3 predicts class 8

        // Create corresponding one-hot encoded labels
        Matrix labels(3, 10);
        labels.fillwith(3, 10, 0.0);
        labels(0,2) = 1.0;  // Sample 1 is class 2
        labels(1,5) = 1.0;  // Sample 2 is class 5
        labels(2,8) = 1.0;  // Sample 3 is class 8

        float acc = compute_accuracy(predictions, labels);
        assert(std::abs(acc - 1.0) < 0.001);
        std::cout << "Test case 1 (100% accuracy) passed!" << std::endl;
    }

    // Test case 2: Mixed predictions (66.67% accuracy)
    {
        Matrix predictions(3, 10);
        predictions.fillwith(3, 10, 0.1);
        predictions(0,2) = 0.9;  // Correct: predicts class 2
        predictions(1,4) = 0.9;  // Wrong: predicts class 4 (should be 5)
        predictions(2,8) = 0.9;  // Correct: predicts class 8

        Matrix labels(3, 10);
        labels.fillwith(3, 10, 0.0);
        labels(0,2) = 1.0;
        labels(1,5) = 1.0;  // Note: true label is 5, but prediction is 4
        labels(2,8) = 1.0;

        float acc = compute_accuracy(predictions, labels);
        assert(std::abs(acc - 0.6667) < 0.001);
        std::cout << "Test case 2 (66.67% accuracy) passed!" << std::endl;
    }

    // Test case 3: All wrong predictions (0% accuracy)
    {
        Matrix predictions(3, 10);
        predictions.fillwith(3, 10, 0.1);
        predictions(0,3) = 0.9;  // Wrong: predicts class 3 (should be 2)
        predictions(1,6) = 0.9;  // Wrong: predicts class 6 (should be 5)
        predictions(2,9) = 0.9;  // Wrong: predicts class 9 (should be 8)

        Matrix labels(3, 10);
        labels.fillwith(3, 10, 0.0);
        labels(0,2) = 1.0;
        labels(1,5) = 1.0;
        labels(2,8) = 1.0;

        float acc = compute_accuracy(predictions, labels);
        assert(std::abs(acc - 0.0) < 0.001);
        std::cout << "Test case 3 (0% accuracy) passed!" << std::endl;
    }
}

int main() {
    test_compute_accuracy();
    // STANDARD, MKL, TILE...
    Matrix::setMulMode(Matrix::MulMode::TILE);
    // std::cout << "Set Matrix Multiplication Mode to " << Matrix::mulMode << std::endl;
    // Create network layers
    std::vector<Layer*> layers;
    layers.push_back(new Linear(784, 128, true, true));    // Larger first hidden layer
    layers.push_back(new Sigmoid());
    layers.push_back(new Linear(128, 10, true, true));     // Output layer
    // layers.push_back(new Sigmoid());
    std::cout << "Successfully create layers" << std::endl;
    // Create network
    Network network(layers);
    std::cout << "Successfully create network" << std::endl;
    // Create optimizer
    SGD optimizer(0.003, 0.9);
    // Adam optimizer(0.001, 0.9, 0.999, 1e-8);
    std::cout << "Successfully create optimizer" << std::endl;
    // Create loss function
    CategoricalCrossentropy loss_fn;
    std::cout << "Successfully create loss function" << std::endl;
    // Load training data
    auto [train_images, train_labels] = load_mnist_data("/home/tri/jin/spaw06j0/MOFramework/data/train-images-idx3-ubyte", 
                                                       "/home/tri/jin/spaw06j0/MOFramework/data/train-labels-idx1-ubyte", 
                                                       60000);
    
    // Load test data
    auto [test_images, test_labels] = load_mnist_data("/home/tri/jin/spaw06j0/MOFramework/data/t10k-images-idx3-ubyte", 
                                                     "/home/tri/jin/spaw06j0/MOFramework/data/t10k-labels-idx1-ubyte", 
                                                     10000);
    std::cout << "Successfully load data" << std::endl;
    check_data(train_images, train_labels, "training");
    check_data(test_images, test_labels, "test");
    // Training parameters
    int epochs = 10;
    int batch_size = 256;
    int num_batches = train_images.getRow() / batch_size;
    std::cout << "Start training" << std::endl;
    // Training loop
    for(int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Epoch " << epoch + 1 << " started" << std::endl;
        float total_loss = 0.0;
        for(int batch = 0; batch < num_batches; batch++) {
            // Get batch data
            Matrix batch_images = train_images.slice(batch * batch_size, (batch + 1) * batch_size);
            Matrix batch_labels = train_labels.slice(batch * batch_size, (batch + 1) * batch_size);

            // Forward pass
            Matrix predictions = network.forward(batch_images);
            Matrix loss = loss_fn(predictions, batch_labels);
            Matrix loss_gradient = loss_fn.backward();
            std::vector<std::vector<Matrix>> layer_gradients = network.backward(loss_gradient);
            optimizer.apply_gradient(network, layer_gradients);
            total_loss += loss.mean();
            if(batch % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                         << ", Batch " << batch << "/" << num_batches << ", Loss: " 
                         << loss.mean() << std::endl;
            }
        }
        std::cout << std::endl;
        total_loss /= num_batches;
        std::cout << "Epoch " << epoch + 1 << " completed. Loss: " << total_loss << std::endl;
        // Evaluate on test set
        Matrix test_predictions = network.forward(test_images);
        float accuracy = compute_accuracy(test_predictions, test_labels);
        std::cout << "Epoch " << epoch + 1 << " completed. Test accuracy: " 
                  << accuracy * 100 << "%" << std::endl;
    }
    
    return 0;
}

