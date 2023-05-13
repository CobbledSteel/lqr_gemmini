# Makefile for building the LQR Riccati Recursion code

# Compiler settings
#CXX = g++
#CXX = riscv64-unknown-linux-gnu-g++ 
CXX = riscv64-unknown-elf-g++ 
CXXFLAGS = -std=c++11 -I/usr/include/eigen3 -I./include/

# Source and build directories
SRC_DIR = ./src
BUILD_DIR = ./build

# Source and target files
SRC = $(SRC_DIR)/lqr_riccati.cpp
TARGET = $(BUILD_DIR)/lqr_riccati_new

# Build rules
all: $(TARGET)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
