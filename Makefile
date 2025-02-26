CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
INCLUDES = -I./function

# Source files (excluding main.cpp and test files)
LIB_SRCS = function/linear.cpp \
           function/network.cpp \
           function/matrix.cpp \
           function/optimizer.cpp \
           function/layer.cpp \
           function/loss.cpp

# Object files
LIB_OBJS = $(LIB_SRCS:.cpp=.o)

# Test source files
# TEST_SRCS = test/testMatrix.cpp \
#             test/testLayer.cpp \
#             test/testLinear.cpp \
#             test/testOptimizer.cpp

# # Test object files
# TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# Main program
MAIN_SRC = main.cpp
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)

# Executables
MAIN_TARGET = mnist_train
# TEST_MATRIX = test_matrix
# TEST_LAYER = test_layer
# TEST_LINEAR = test_linear
# TEST_OPTIMIZER = test_optimizer

# # Default target
# all: tests $(MAIN_TARGET)

# # Build and run all tests
# tests: $(TEST_MATRIX) $(TEST_LAYER) $(TEST_LINEAR) $(TEST_OPTIMIZER)
# 	@echo "Running Matrix tests..."
# 	./$(TEST_MATRIX)
# 	@echo "Running Layer tests..."
# 	./$(TEST_LAYER)
# 	@echo "Running Linear tests..."
# 	./$(TEST_LINEAR)
# 	@echo "Running Optimizer tests..."
# 	./$(TEST_OPTIMIZER)

# Main program
$(MAIN_TARGET): $(MAIN_OBJ) $(LIB_OBJS)
	$(CXX) $^ -o $@

# Test executables
# $(TEST_MATRIX): test/testMatrix.o $(LIB_OBJS)
# 	$(CXX) $^ -o $@

# $(TEST_LAYER): test/testLayer.o $(LIB_OBJS)
# 	$(CXX) $^ -o $@

# $(TEST_LINEAR): test/testLinear.o $(LIB_OBJS)
# 	$(CXX) $^ -o $@

# $(TEST_OPTIMIZER): test/testOptimizer.o $(LIB_OBJS)
# 	$(CXX) $^ -o $@

# Compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Dependencies
main.o: main.cpp
function/linear.o: function/linear.cpp function/linear.h
function/network.o: function/network.cpp function/network.h
function/matrix.o: function/matrix.cpp function/matrix.h
function/optimizer.o: function/optimizer.cpp function/optimizer.h
function/layer.o: function/layer.cpp function/layer.h
function/loss.o: function/loss.cpp function/loss.h

# Clean
clean:
	rm -f $(LIB_OBJS) $(TEST_OBJS) $(MAIN_OBJ) $(MAIN_TARGET) $(TEST_MATRIX) $(TEST_LAYER) $(TEST_LINEAR) $(TEST_OPTIMIZER)

.PHONY: all clean tests 