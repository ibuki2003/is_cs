#pragma once

#include "config.hpp"
#include <cstdlib>

struct Matrix {
  size_t n;
  double *data;

  Matrix(size_t n): n(n), data((double*)aligned_alloc(32, n * n * sizeof(double))) {}
  Matrix(Matrix &&other): n(other.n), data(other.data) { other.data = nullptr; }
  ~Matrix() { free(data); }

  inline double* operator[] (size_t i) { return data + n * i; }
  inline double const* operator[] (size_t i) const { return data + n * i; }

  // deny copy and assignment
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
};
