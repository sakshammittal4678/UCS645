# Compiler
CXX = g++

# Compiler flags
# -std=c++11 : C++11 standard
# -Wall       : all warnings
# -O3         : max optimization (enables auto-vectorization)
# -fopenmp    : OpenMP support
# -march=native: use all CPU instruction sets (AVX2, SSE4, etc.)
CXXFLAGS = -std=c++11 -Wall -O3 -fopenmp -march=native

# Target executable
TARGET = correlate

# Source files
SOURCES = main.cpp correlate.cpp

# Object files (auto-derived from sources)
OBJECTS = $(SOURCES:.cpp=.o)

# Header files
HEADERS = correlate.h

# -------------------------------------------------------
# Default target
# -------------------------------------------------------
all: $(TARGET)

# Link object files
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile each .cpp to .o
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -------------------------------------------------------
# Run targets (Part 4: vary matrix size from command line)
# -------------------------------------------------------
# Default run: 500x1000 matrix, max threads
run: $(TARGET)
	./$(TARGET) 500 1000

# Run with custom size: make run-custom NY=200 NX=500
run-custom: $(TARGET)
	./$(TARGET) $(NY) $(NX) $(THREADS)

# Benchmark: vary matrix size
bench: $(TARGET)
	@echo "=== Benchmark: varying matrix size ==="
	@echo "--- 100 x 500 ---"
	./$(TARGET) 100 500
	@echo "--- 300 x 1000 ---"
	./$(TARGET) 300 1000
	@echo "--- 500 x 2000 ---"
	./$(TARGET) 500 2000

# Benchmark: vary number of threads (matrix fixed at 400x1000)
bench-threads: $(TARGET)
	@echo "=== Benchmark: varying thread count ==="
	@echo "--- 1 thread ---"
	./$(TARGET) 400 1000 1
	@echo "--- 2 threads ---"
	./$(TARGET) 400 1000 2
	@echo "--- 4 threads ---"
	./$(TARGET) 400 1000 4
	@echo "--- 8 threads ---"
	./$(TARGET) 400 1000 8

# -------------------------------------------------------
# perf stat profiling (Part 4)
# Run: make perf-seq  or  make perf-par
# -------------------------------------------------------
perf-seq: $(TARGET)
	perf stat -e cycles,instructions,cache-misses,cache-references ./$(TARGET) 200 500 1

perf-par: $(TARGET)
	perf stat -e cycles,instructions,cache-misses,cache-references ./$(TARGET) 200 500

# ---- Fix perf permissions (run once if perf gives errors) ----
fix-perf:
	sudo sysctl kernel.perf_event_paranoid=1

# -------------------------------------------------------
# Clean
# -------------------------------------------------------
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all run run-custom bench bench-threads perf-seq perf-par clean
