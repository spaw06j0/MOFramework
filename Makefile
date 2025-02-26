# Python and pybind11 configuration
PYTHON_CONFIG = python3-config
PYTHON_INCLUDE = $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags)
PYBIND11_INCLUDE = $(shell python3 -m pybind11 --includes)

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -fPIC
INCLUDES = -I./ $(PYBIND11_INCLUDE)

# Source directories
SRCDIR = function
OBJDIR = obj

# Create object directory if it doesn't exist
$(shell mkdir -p $(OBJDIR))

# Source files
LIB_SRCS = $(SRCDIR)/linear.cpp \
           $(SRCDIR)/network.cpp \
           $(SRCDIR)/matrix.cpp \
           $(SRCDIR)/optimizer.cpp \
           $(SRCDIR)/layer.cpp \
           $(SRCDIR)/loss.cpp \
           $(SRCDIR)/activation.cpp

# Object files with path adjustment
LIB_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(LIB_SRCS))

BINDING_SRC = $(SRCDIR)/binding.cpp
BINDING_OBJ = $(OBJDIR)/binding.o
MODULE_NAME = pynet
MODULE_FILE = $(MODULE_NAME)$(shell python3-config --extension-suffix)

# Main program
MAIN_SRC = main.cpp
MAIN_OBJ = $(OBJDIR)/main.o

# Executables
MAIN_TARGET = mnist_train

# Default target
all: $(MAIN_TARGET)

python_module: $(MODULE_FILE)

# Make sure object directory exists
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Main program
$(MAIN_TARGET): $(MAIN_OBJ) $(LIB_OBJS)
	$(CXX) $^ -o $@ $(shell $(PYTHON_CONFIG) --ldflags)

# Python module target 
$(MODULE_FILE): $(BINDING_SRC) $(LIB_OBJS)
	$(CXX) -shared -fPIC $(INCLUDES) $^ -o $@ $(PYTHON_LDFLAGS)

# Compilation rule for main.cpp
$(OBJDIR)/main.o: $(MAIN_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compilation rule for binding.cpp
$(OBJDIR)/binding.o: $(BINDING_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o

# Compilation rule for source files in function directory
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJDIR) $(MAIN_TARGET) $(MODULE_FILE)

.PHONY: all clean