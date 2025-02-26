#include "matrix.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_matrix() {
    // Test Constructor
    Matrix mat1(2, 3);
    assert(mat1.row == 2);
    assert(mat1.col == 3);

    // Test fillwith
    mat1 = Matrix::fillwith(2, 3, 2.5);
    for (size_t i = 0; i < mat1.row; i++) {
        for (size_t j = 0; j < mat1.col; j++) {
            assert(mat1(i, j) == 2.5);
        }
    }

    // Test copy constructor
    Matrix mat2(mat1);
    assert(mat2 == mat1);

    // Test assignment operator
    Matrix mat3;
    mat3 = mat1;
    assert(mat3 == mat1);

    // Test addtion
    Matrix mat4 = Matrix::fillwith(2, 3, 1.0);
    Matrix mat5 = mat1 + mat4;
    for (size_t i = 0; i < mat5.row; i++) {
        for (size_t j = 0; j < mat5.col; j++) {
            assert(mat5(i, j) == 3.5);
        }
    }
    
    // Test scalar addtion
    Matrix mat6 = mat1 + 1.0;
    for (size_t i = 0; i < mat6.row; i++) {
        for (size_t j = 0; j < mat6.col; j++) {
            assert(mat6(i, j) == 3.5);
        }
    }

    // Test +=
    mat1 += mat4;
    for (size_t i = 0; i < mat1.row; i++) {
        for (size_t j = 0; j < mat1.col; j++) {
            assert(mat1(i, j) == 3.5);
        }
    }
    
    // Test -=
    mat1 -= mat4;
    for (size_t i = 0; i < mat1.row; i++) {
        for (size_t j = 0; j < mat1.col; j++) {
            assert(mat1(i, j) == 2.5);
        }
    }

    // Test -= scalar
    mat1 -= 1.0;
    for (size_t i = 0; i < mat1.row; i++) {
        for (size_t j = 0; j < mat1.col; j++) {
            assert(mat1(i, j) == 1.5);
        }
    }
    
    // Test *=
    Matrix mat7 = Matrix::fillwith(2, 3, 2.0);
    Matrix mat8 = Matrix::fillwith(2, 3, 2.0);
    mat8 *= mat7;
    for (size_t i = 0; i < mat8.row; i++) {
        for (size_t j = 0; j < mat8.col; j++) {
            assert(mat8(i, j) == 4.0);
        }
    }

    // Test *= scalar
    mat8 *= 2.0;
    for (size_t i = 0; i < mat8.row; i++) {
        for (size_t j = 0; j < mat8.col; j++) {
            assert(mat8(i, j) == 8.0);
        }
    }

    // Test /=
    Matrix mat9 = Matrix::fillwith(2, 3, 2.0);
    mat9 /= mat7;
    for (size_t i = 0; i < mat9.row; i++) {
        for (size_t j = 0; j < mat9.col; j++) {
            assert(mat9(i, j) == 1.0);
        }
    }

    // Test /= scalar
    mat9 /= 2.0;
    for (size_t i = 0; i < mat9.row; i++) {
        for (size_t j = 0; j < mat9.col; j++) {
            assert(mat9(i, j) == 0.5);
        }
    }

    // Test power
    Matrix mat10 = Matrix::fillwith(2, 3, 2.0);
    Matrix mat11 = mat10.power(2.0);
    for (size_t i = 0; i < mat11.row; i++) {
        for (size_t j = 0; j < mat11.col; j++) {
            assert(mat11(i, j) == 4.0);
        }
    }

    // Test exp
    Matrix mat12 = Matrix::fillwith(2, 3, 1.0);
    Matrix mat13 = mat12.exp();
    for (size_t i = 0; i < mat13.row; i++) {
        for (size_t j = 0; j < mat13.col; j++) {
            assert(std::abs(mat13(i, j) == std::exp(1.0)));
        }
    }

    // Test log
    Matrix mat14 = Matrix::fillwith(2, 3, std::exp(1.0));
    Matrix mat15 = mat14.log();
    assert(mat15(0, 0) == 1.0);
    assert(mat15(0, 1) == 1.0);
    assert(mat15(0, 2) == 1.0);

    // Test sigmoid
    Matrix mat16 = Matrix::fillwith(2, 3, 0.0);
    Matrix mat17 = mat16.sigmoid();
    assert(mat17(0, 0) == 0.5);
    assert(mat17(0, 1) == 0.5);
    assert(mat17(0, 2) == 0.5);

    std::cout << "All tests passed!" << std::endl;
}

void test_matrix_edge_cases() {
    // Test empty matrix
    Matrix empty;
    assert(empty.row == 0 && empty.col == 0 && empty.data == nullptr);

    // Test zero-sized dimensions
    // try {
    //     Matrix invalid(0, 5);
    //     assert(false && "Should throw exception for zero rows");
    // } catch (const std::runtime_error&) {}

    // try {
    //     Matrix invalid(5, 0);
    //     assert(false && "Should throw exception for zero columns");
    // } catch (const std::runtime_error&) {}

    // Test matrix broadcasting edge cases
    Matrix mat1 = Matrix::fillwith(1, 3, 1.0);
    Matrix mat2 = Matrix::fillwith(3, 1, 2.0);
    
    // Broadcasting 1x3 with 3x1 should give 3x3
    Matrix result = mat1 + mat2;
    assert(result.row == 3 && result.col == 3);
    
    // Test division by zero
    Matrix mat3 = Matrix::fillwith(2, 2, 0.0);
    try {
        Matrix result = mat1 / mat3;
        assert(false && "Should throw exception for division by zero");
    } catch (const std::runtime_error&) {}

    // Test numerical stability for exp/log
    Matrix large(1, 1);
    large(0,0) = 1000.0;  // Very large number
    Matrix small(1, 1);
    small(0,0) = 1e-300;  // Very small number
    
    Matrix exp_result = large.exp();
    Matrix log_result = small.log();
    // assert(!std::isinf(exp_result(0,0)) && !std::isnan(exp_result(0,0)));
    // assert(!std::isinf(log_result(0,0)) && !std::isnan(log_result(0,0)));

    // Test matrix multiplication dimensions
    Matrix m1(2, 3);
    Matrix m2(4, 2);
    try {
        Matrix result = multiply(m1, m2);
        assert(false && "Should throw exception for incompatible dimensions");
    } catch (const std::runtime_error&) {}

    // Test slice edge cases
    Matrix mat4(5, 2);
    try {
        Matrix slice = mat4.slice(3, 2);  // end < start
        assert(false && "Should throw exception for invalid slice range");
    } catch (const std::runtime_error&) {}

    try {
        Matrix slice = mat4.slice(0, 6);  // end > rows
        assert(false && "Should throw exception for out of bounds slice");
    } catch (const std::runtime_error&) {}

    // Test T() function
    Matrix mat5(2, 3);
    mat5(0, 0) = 1.0;
    mat5(0, 1) = 2.0;
    mat5(0, 2) = 3.0;
    mat5(1, 0) = 4.0;
    mat5(1, 1) = 5.0;
    mat5(1, 2) = 6.0;
    Matrix transposed = mat5.T();
    assert(transposed.row == 3 && transposed.col == 2);
    assert(transposed(0, 0) == 1.0);
    assert(transposed(0, 1) == 4.0);
    assert(transposed(1, 0) == 2.0);
    assert(transposed(1, 1) == 5.0);
    assert(transposed(2, 0) == 3.0);
    assert(transposed(2, 1) == 6.0);

    // Test hadamard product edge cases
    // Matrix mat5(2, 3);
    // Matrix mat6(2, 2);
    // try {
    //     Matrix result = mat5.hadamard(mat6);
    //     assert(false && "Should throw exception for mismatched dimensions");
    // } catch (const std::runtime_error&) {}
}

int main() {
    try {
        test_matrix();
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    try {
        test_matrix_edge_cases();
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}