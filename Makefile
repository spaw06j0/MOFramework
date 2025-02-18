CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
INCLUDES = -I./function

# Source files
SRCS = main.cpp \
       function/linear.cpp \
       function/network.cpp \
       function/matrix.cpp \
       function/optimizer.cpp \
       function/layer.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = mnist_train

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

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

# Clean
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean 