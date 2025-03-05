# Python and pybind11 configuration
PYTHON_CONFIG = python3-config
PYTHON_INCLUDE = $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags)
PYBIND11_INCLUDE = $(shell python3 -m pybind11 --includes)

# MKL configuration for Conda
CONDA_PREFIX ?= $(shell conda info --env | grep -i '*' | awk '{print $$3}')
MKL_INCLUDE = -I$(CONDA_PREFIX)/include
# Simplified MKL linking
MKL_LIB = -L$(CONDA_PREFIX)/lib -lmkl_rt -lm -ldl

# CUDA configuration
CUDA_PATH ?= /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcuda
NVCC = $(CUDA_PATH)/bin/nvcc
NVCCFLAGS = -std=c++14 -O2 -Xcompiler -fPIC

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -fPIC -m64 -fopenmp -pthread
INCLUDES = -I./ $(PYBIND11_INCLUDE) $(MKL_INCLUDE) $(CUDA_INCLUDE)
LDFLAGS = $(PYTHON_LDFLAGS) $(MKL_LIB) $(CUDA_LIB)

# Source directories
SRCDIR = function
OBJDIR = obj
CUDADIR = cuda

# Create object directory if it doesn't exist
$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(CUDADIR))

# Source files
LIB_SRCS = $(SRCDIR)/linear.cpp \
           $(SRCDIR)/network.cpp \
           $(SRCDIR)/matrix.cpp \
           $(SRCDIR)/optimizer.cpp \
           $(SRCDIR)/layer.cpp \
           $(SRCDIR)/loss.cpp \
           $(SRCDIR)/activation.cpp
# CUDA source files
CUDA_SRCS = $(CUDADIR)/matrix_cuda.cu

# Object files with path adjustment
LIB_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(LIB_SRCS))
CUDA_OBJS = $(OBJDIR)/matrix_cuda.o

BINDING_SRC = $(SRCDIR)/binding.cpp
MODULE_NAME = pynet
MODULE_FILE = $(MODULE_NAME)$(shell python3-config --extension-suffix)

# Main program
MAIN_SRC = main.cpp
MAIN_OBJ = $(OBJDIR)/main.o

# Performance test files
TEST_PERF_SRC = test/testPerformance.cpp
TEST_PERF_OBJ = $(OBJDIR)/testperformance.o
TEST_PERF_TARGET = testperformance

# Executables
MAIN_TARGET = mnist_train

# Default target
all: $(MAIN_TARGET)

python_module: $(MODULE_FILE)

# Make sure object directory exists
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Main program target
$(MAIN_TARGET): $(MAIN_OBJ) $(LIB_OBJS) $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Python module target 
$(MODULE_FILE): $(BINDING_SRC) $(LIB_OBJS) $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Performance test program
$(TEST_PERF_TARGET): $(TEST_PERF_OBJ) $(OBJDIR)/matrix.o $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compilation rule for main.cpp
$(OBJDIR)/main.o: $(MAIN_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compilation rule for performance test
$(OBJDIR)/testperformance.o: $(TEST_PERF_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compilation rule for source files in function directory
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compilation rule for CUDA source files
$(OBJDIR)/matrix_cuda.o: $(CUDA_SRCS) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

test: $(TEST_PERF_TARGET)
	./$(TEST_PERF_TARGET)
    
# Clean
clean:
	rm -rf $(OBJDIR) $(MAIN_TARGET) $(MODULE_FILE)

.PHONY: all clean