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
    mat1.fillwith(2, 3, 2.5);
    assert(mat1(0, 0) == 2.5);
    assert(mat1(0, 1) == 2.5);
    assert(mat1(0, 2) == 2.5);
    assert(mat1(1, 0) == 2.5);
    assert(mat1(1, 1) == 2.5);
    assert(mat1(1, 2) == 2.5);

    // Test copy constructor
    Matrix mat2(mat1);
    assert(mat2 == mat1);

    // Test assignment operator
    Matrix mat3;
    mat3 = mat1;
    assert(mat3 == mat1);

    // Test addtion
    Matrix mat4(2, 3);
    mat4.fillwith(2, 3, 1.0);
    Matrix mat5 = mat1 + mat4;
    assert(mat5(0, 0) == 3.5);
    assert(mat5(0, 1) == 3.5);
    assert(mat5(0, 2) == 3.5);
    
    // Test scalar addtion
    Matrix mat6 = mat1 + 1.0;
    assert(mat6(0, 0) == 3.5);
    assert(mat6(0, 1) == 3.5);
    assert(mat6(0, 2) == 3.5);

    // Test +=
    mat1 += mat4;
    assert(mat1(0, 0) == 3.5);
    assert(mat1(0, 1) == 3.5);
    assert(mat1(0, 2) == 3.5);
    
    // Test -=
    mat1 -= mat4;
    assert(mat1(0, 0) == 2.5);
    assert(mat1(0, 1) == 2.5);
    assert(mat1(0, 2) == 2.5);

    // Test -= scalar
    mat1 -= 1.0;
    assert(mat1(0, 0) == 1.5);
    assert(mat1(0, 1) == 1.5);
    assert(mat1(0, 2) == 1.5);
    
    // Test *=
    Matrix mat7(2, 3);
    mat7.fillwith(2, 3, 2.0);
    Matrix mat8(2, 3);
    mat8.fillwith(2, 3, 2.0);
    mat8 *= mat7;
    assert(mat8(0, 0) == 4.0);
    assert(mat8(0, 1) == 4.0);
    assert(mat8(0, 2) == 4.0);

    // Test *= scalar
    mat8 *= 2.0;
    assert(mat8(0, 0) == 8.0);
    assert(mat8(0, 1) == 8.0);
    assert(mat8(0, 2) == 8.0);

    // Test /=
    Matrix mat9(2, 3);
    mat9.fillwith(2, 3, 2.0);
    mat9 /= mat7;
    assert(mat9(0, 0) == 1.0);

    // Test /= scalar
    mat9 /= 2.0;
    assert(mat9(0, 0) == 0.5);
    assert(mat9(0, 1) == 0.5);
    assert(mat9(0, 2) == 0.5);

    // Test power
    Matrix mat10(2, 3);
    mat10.fillwith(2, 3, 2.0);
    Matrix mat11 = mat10.power(2.0);
    assert(mat11(0, 0) == 4.0);
    assert(mat11(0, 1) == 4.0);
    assert(mat11(0, 2) == 4.0);

    // Test exp
    Matrix mat12(2, 3);
    mat12.fillwith(2, 3, 1.0);
    Matrix mat13 = mat12.exp();
    assert(mat13(0, 0) == std::exp(1.0));
    assert(mat13(0, 1) == std::exp(1.0));
    assert(mat13(0, 2) == std::exp(1.0));

    // Test log
    Matrix mat14(2, 3);
    mat14.fillwith(2, 3, std::exp(1.0));
    Matrix mat15 = mat14.log();
    assert(mat15(0, 0) == 1.0);
    assert(mat15(0, 1) == 1.0);
    assert(mat15(0, 2) == 1.0);

    // Test sigmoid
    Matrix mat16(2, 3);
    mat16.fillwith(2, 3, 0.0);
    Matrix mat17 = mat16.sigmoid();
    assert(mat17(0, 0) == 0.5);
    assert(mat17(0, 1) == 0.5);
    assert(mat17(0, 2) == 0.5);

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    try {
        test_matrix();
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}