#include "prod_blas.hpp"

#include <cassert>
#include <openblas/cblas.h>

void prod_blas(const Matrix& a, const Matrix& b, Matrix& c) {
  assert(a.n == b.n && b.n == c.n);
  const size_t n = a.n;
  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    n,
    n,
    n,
    1.0,
    a[0],
    n,
    b[0],
    n,
    0.0,
    c[0],
    n
  );

}
