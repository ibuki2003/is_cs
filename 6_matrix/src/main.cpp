#include <iostream>
#include <chrono>
#include <prod_blas.hpp>
#include <random>
#include "matrix.hpp"
#include "prod_naive.hpp"
#include "prod_fast.hpp"

constexpr const size_t ITERATIONS = 1;
constexpr double EPSILON = 1e-6;

MatrixA a, b, c, d;

int main() {

  // preparation
  std::mt19937 gen(42); // seed is constant
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  for (size_t i = 0; i < MATRIX_SIZE; ++i) {
    for (size_t j = 0; j < MATRIX_SIZE; ++j) {
      a[i][j] = dis(gen);
      b[i][j] = dis(gen);
    }
  }

  // execution

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    prod_fast(a, b, c);
  }
  auto end = std::chrono::high_resolution_clock::now();

  long long operations = MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE * 2 * ITERATIONS;

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
  std::cout << "Performance: " << operations / elapsed_seconds.count() / 1e9 << " GFLOPS\n";

  // validation
  auto start2 = std::chrono::high_resolution_clock::now();
  prod_blas(a, b, d);
  auto end2 = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end2 - start2;
  std::cout << "Elapsed time (OpenBLAS): " << elapsed_seconds.count() << "s\n";
  std::cout << "Performance: " << operations / elapsed_seconds.count() / 1e9 << " GFLOPS\n";

  for (size_t i = 0; i < MATRIX_SIZE; ++i) {
    for (size_t j = 0; j < MATRIX_SIZE; ++j) {
      if (std::abs(c[i][j] - d[i][j]) > EPSILON) {
        std::cerr << "Test failed\n";
        std::cerr << "c[" << i << "][" << j << "] = " << c[i][j] << " (expected " << d[i][j] << ")\n";
        return 1;
      }
    }
  }
  std::cout << "Test passed\n";
}

