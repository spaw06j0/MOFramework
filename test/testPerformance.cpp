#include "../function/matrix.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

template<typename Func>
double measureTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

Matrix createRandomMatrix(size_t rows, size_t cols) {
    Matrix m(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            m(i, j) = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    return m;
}

bool matricesEqual(const Matrix& a, const Matrix& b, double tolerance = 1e-6) {
    if (a.getRow() != b.getRow() || a.getCol() != b.getCol()) {
        return false;
    }
    
    for (size_t i = 0; i < a.getRow(); i++) {
        for (size_t j = 0; j < a.getCol(); j++) {
            if (std::abs(a(i, j) - b(i, j)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

void testPerformance(size_t m, size_t n, size_t k) {
    std::cout << "\nTesting matrices of size: " << m << "x" << k << " * " << k << "x" << n << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    
    Matrix A = createRandomMatrix(m, k);
    Matrix B = createRandomMatrix(k, n);
    
    // Multiply using standard algorithm for reference
    double standardTime = measureTime([&]() {
        multiply(A, B);
    });
    
    std::vector<std::pair<std::string, double>> results; // {method name, cost time}
    results.push_back({"Standard", standardTime});
    
    // Test MKL implementation
    Matrix C_mkl;
    double mklTime = measureTime([&]() {
        C_mkl = multiply_mkl(A, B);
    });
    results.push_back({"MKL", mklTime});
    
    // Test tiled implementation with different tile sizes
    std::vector<int> tileSizes = {8, 16, 32, 64};
    for (int tileSize : tileSizes) {
        Matrix C_tile;
        double tileTime = measureTime([&]() {
            C_tile = multiply_tile(A, B, tileSize);
        });
        
        // Verify result matches the standard implementation
        Matrix C_std = multiply(A, B);
        bool correct = matricesEqual(C_std, C_tile);
        
        results.push_back({"Tiled (" + std::to_string(tileSize) + ")", tileTime});
        if (!correct) {
            std::cout << "Warning: Tiled implementation with size " << tileSize << " produced incorrect results!" << std::endl;
        }
    }
    
    // Print results
    std::cout << std::setw(15) << "Method" << std::setw(15) << "Time (ms)" << std::setw(15) << "Speedup" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(15) << result.first 
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.second 
                  << std::setw(15) << std::fixed << std::setprecision(2) << (standardTime / result.second)
                  << std::endl;
    }
}

int main() {
    // Seed random number generator
    srand(42);
    
    std::cout << "Matrix Multiplication Performance Test" << std::endl;
    std::cout << "=====================================" << std::endl;
    // Test different matrix sizes
    std::vector<std::tuple<size_t, size_t, size_t>> testSizes = {
        {64, 64, 64},        // Small
        {128, 128, 128},     // Medium
        {512, 512, 512},     // Large
        {1024, 1024, 1024},  // Very large
        {1000, 100, 1000},
        {2000, 500, 50}
    };

    for (const auto& [m, k, n] : testSizes) {
        testPerformance(m, n, k);
    }

    return 0;
}