#include "prod_naive.hpp"
#include <cassert>
#include <cstring>

void prod_naive(const Matrix& a, const Matrix& b, Matrix& c) {
  assert(a.n == b.n && b.n == c.n);
  const size_t n = a.n;
  std::memset(c.data, 0, sizeof(Matrix));

  for (size_t k = 0; k < n; ++k) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}
