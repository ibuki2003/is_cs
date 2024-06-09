#include "prod_naive.hpp"
#include <cstring>

void prod_naive(const MatrixA& a, const MatrixA& b, MatrixA& c) {
  std::memset(c.data, 0, sizeof(MatrixA));

  for (size_t k = 0; k < MATRIX_SIZE; ++k) {
    for (size_t i = 0; i < MATRIX_SIZE; ++i) {
      for (size_t j = 0; j < MATRIX_SIZE; ++j) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}
