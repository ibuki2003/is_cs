#include "prod_blas.hpp"

#include <openblas/cblas.h>

void prod_blas(const MatrixA& a, const MatrixA& b, MatrixA& c) {
  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    MATRIX_SIZE,
    MATRIX_SIZE,
    MATRIX_SIZE,
    1.0,
    a.data[0],
    MATRIX_SIZE,
    b.data[0],
    MATRIX_SIZE,
    0.0,
    c.data[0],
    MATRIX_SIZE
  );

}
