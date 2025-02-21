#include <iostream>
#include <cassert>
#include "../function/matrix.h"

// Function to test
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
        assert(acc == 1.0);
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
        assert(acc == 0.0);
        std::cout << "Test case 3 (0% accuracy) passed!" << std::endl;
    }
}

int main() {
    try {
        test_compute_accuracy();
        std::cout << "All accuracy tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 