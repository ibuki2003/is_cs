#include <cassert>
#include <iostream>
#include <chrono>
#include <prod_blas.hpp>
#include <random>
#include "matrix.hpp"
#include "prod_naive.hpp"
#include "prod_fast.hpp"

constexpr const size_t ITERATIONS = 1;
constexpr double EPSILON = 1e-6;


std::mt19937 gen{42};
void randomize(Matrix& a) {
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  for (size_t i = 0; i < a.n; ++i) {
    for (size_t j = 0; j < a.n; ++j) {
      a[i][j] = dis(gen);
    }
  }
}

enum mode { NAIVE, FAST, BLAS };

double measure_performance(size_t n, mode m) {
  Matrix a(n), b(n), c(n), d(n);

  // execution

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    switch (m) {
      case NAIVE: prod_naive(a, b, c); break;
      case FAST: prod_fast(a, b, c); break;
      case BLAS: prod_blas(a, b, c); break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  long long operations = n * n * n * 2 * ITERATIONS;

  std::chrono::duration<double> elapsed_seconds = end - start;
  return operations / elapsed_seconds.count() / 1e9; // GFLOPS
}

bool validate(size_t n) {
  Matrix a(n), b(n), c(n), d(n);
  randomize(a);
  randomize(b);

  prod_fast(a, b, c);
  prod_blas(a, b, d);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (std::abs(c[i][j] - d[i][j]) > EPSILON) {
        std::cerr << "c[" << i << "][" << j << "] = " << c[i][j] << " (expected " << d[i][j] << ")\n";
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char** argv) {
  mode m = FAST;
  if (argc > 1) {
    switch (tolower(argv[1][0])) {
      case 'n': m = NAIVE; break;
      case 'f': m = FAST; break;
      case 'b': m = BLAS; break;
    }
  }

  if (m == FAST) {

    if (validate(1024)) {
      std::cout << "Test passed\n";
    } else {
      std::cout << "Test failed\n";
      return 1;
    }
  }

  for (size_t n = 256; n <= 4096; n += 256) {
    std::cout << n << '\t' << measure_performance(n, m) << '\n';
  }

}

