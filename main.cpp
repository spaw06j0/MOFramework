#include "network.h"
#include "linear.h"
#include "activation.h"
#include "optimizer.h"
#include "loss.h"
#include <vector>
#include <iostream>
#include <random>
#include <fstream>

// Helper function to load MNIST data (you'll need to implement this)
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
            images(i, j) = static_cast<float>(pixel) / 255.0;  // Normalize to [0,1]
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

    return {images, labels};
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
            if(labels(i,j) > 0.5) {
                true_idx = j;
                break;
            }
        }
        
        if(pred_idx == true_idx) correct++;
    }
    
    return float(correct) / total;
}

int main() {
    // Create network layers
    std::vector<Layer*> layers;
    layers.push_back(new Linear(784, 128, true));  // Input layer: 784 -> 128
    layers.push_back(new ReLU());                  // ReLU activation
    layers.push_back(new Linear(128, 10, true));   // Output layer: 128 -> 10 (for digits 0-9)
    
    // Create network
    Network network(layers);
    
    // Create optimizer
    SGD optimizer(0.01, 0.9);  // learning rate = 0.01, momentum = 0.9

    // Create loss function
    CategoricalCrossentropy loss_fn;

    // Load training data
    auto [train_images, train_labels] = load_mnist_data("/home/tri/jin/spaw06j0/MOFramework/data/train-images-idx3-ubyte", 
                                                       "/home/tri/jin/spaw06j0/MOFramework/data/train-labels-idx1-ubyte", 
                                                       60000);
    
    // Load test data
    auto [test_images, test_labels] = load_mnist_data("/home/tri/jin/spaw06j0/MOFramework/data/t10k-images-idx3-ubyte", 
                                                     "/home/tri/jin/spaw06j0/MOFramework/data/t10k-labels-idx1-ubyte", 
                                                     10000);
    
    // Training parameters
    int epochs = 10;
    int batch_size = 32;
    int num_batches = train_images.getRow() / batch_size;
    
    // Training loop
    for(int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        
        for(int batch = 0; batch < num_batches; batch++) {
            // Get batch data
            Matrix batch_images = train_images.slice(batch * batch_size, (batch + 1) * batch_size);
            Matrix batch_labels = train_labels.slice(batch * batch_size, (batch + 1) * batch_size);
            
            // Forward pass
            Matrix predictions = network.forward(batch_images);
            
            // Compute loss
            Matrix loss = loss_fn(predictions, batch_labels);
            total_loss += loss.sum();
            
            // Get initial gradient from loss function
            Matrix loss_gradient = loss_fn.backward();
            
            // Backward pass through network - this returns gradients for each layer
            std::vector<std::vector<Matrix>> layer_gradients = network.backward(loss_gradient);
            
            // Apply gradients using optimizer
            optimizer.apply_gradient(network, layer_gradients);
            
            // Print progress
            if(batch % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                         << ", Batch " << batch << "/" << num_batches << std::endl;
            }
        }
        total_loss /= num_batches;
        std::cout << "Epoch " << epoch + 1 << " completed. Loss: " << total_loss << std::endl;
        // Evaluate on test set
        Matrix test_predictions = network.forward(test_images);
        float accuracy = compute_accuracy(test_predictions, test_labels);
        std::cout << "Epoch " << epoch + 1 << " completed. Test accuracy: " 
                  << accuracy * 100 << "%" << std::endl;
    }
    
    // Clean up
    for(auto layer : layers) {
        delete layer;
    }
    
    return 0;
}

