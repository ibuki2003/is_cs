#include "prod_fast.hpp"
#include <cassert>
#include <config.hpp>
#include <cstring>
#include <immintrin.h>

constexpr size_t NB = 256;
constexpr size_t MB = 128;

__align_avx__ double abuf[MB/4][NB][4];
/* __align_avx__ double bbuf[NB][4]; */
__align_avx__ __m256d bbuf[NB];
__align_avx__ double cbuf[MB][4];


inline void prod44(
  __m256d a,
  __m256d b,
  __m256d& s0,
  __m256d& s1,
  __m256d& s2,
  __m256d& s3
) {
  s0 = _mm256_fmadd_pd(a, b, s0);
  b = _mm256_permute_pd(b, 0b0101); // swap each pair
  s1 = _mm256_fmadd_pd(a, b, s1);
  b = _mm256_permute2f128_pd(b, b, 0x01); // swap high and low
  s3 = _mm256_fmadd_pd(a, b, s3);
  b = _mm256_permute_pd(b, 0b0101); // swap each pair
  s2 = _mm256_fmadd_pd(a, b, s2);
}

inline void shuf44(
    __m256d& s0,
    __m256d& s1,
    __m256d& s2,
    __m256d& s3
) {
  __m256d t0 = _mm256_shuffle_pd(s0, s1, 0b0000);
  __m256d t1 = _mm256_shuffle_pd(s1, s0, 0b1111);
  __m256d t2 = _mm256_shuffle_pd(s2, s3, 0b0000);
  __m256d t3 = _mm256_shuffle_pd(s3, s2, 0b1111);
  t2 = _mm256_permute2f128_pd(t2, t2, 0x01);
  t3 = _mm256_permute2f128_pd(t3, t3, 0x01);

  s2 = _mm256_blend_pd(t0, t2, 0b0011);
  s0 = _mm256_blend_pd(t0, t2, 0b1100);
  s3 = _mm256_blend_pd(t1, t3, 0b0011);
  s1 = _mm256_blend_pd(t1, t3, 0b1100);
}


void prod_fast(const Matrix& a, const Matrix& b, Matrix& c) {
  assert(a.n == b.n && b.n == c.n);
  const size_t n = a.n;

  memset(c.data, 0, sizeof(double) * n * n);

  for (size_t i = 0; i < n; i += NB) {
    for (size_t j = 0; j < n; j += MB) {
      // copy to abuf
      // abuf = a_j,i
      for (size_t k = 0; k < MB; k += 4) {
        for (size_t l = 0; l < NB; ++l) {
          abuf[k/4][l][0] = a[j+k+0][i+l];
          abuf[k/4][l][1] = a[j+k+1][i+l];
          abuf[k/4][l][2] = a[j+k+2][i+l];
          abuf[k/4][l][3] = a[j+k+3][i+l];
        }
      }

      for (size_t k = 0; k < n; k += 4) {
        memset(cbuf, 0, sizeof(cbuf));
        __m256d *const cbuf0 = (__m256d*)cbuf;

        // copy to bbuf
        // bbuf = b_i,k
        for (size_t l = 0; l < NB; ++l) {
          bbuf[l] = *(__m256d*)(b[i+l] + k);
        }

        for (size_t l = 0; l < MB; l += 4) {
          __m256d *const cbufi = cbuf0 + l;
          __m256d const *const abufil = (__m256d const*)(abuf + l/4);

          __m256d s0 = _mm256_setzero_pd();
          __m256d s1 = _mm256_setzero_pd();
          __m256d s2 = _mm256_setzero_pd();
          __m256d s3 = _mm256_setzero_pd();

          for (size_t m = 0; m < NB; ++m) {
            prod44(abufil[m], bbuf[m], s0, s1, s2, s3);
          }
          shuf44(s0, s1, s2, s3);
          cbufi[0] = _mm256_add_pd(cbufi[0], s0);
          cbufi[1] = _mm256_add_pd(cbufi[1], s1);
          cbufi[2] = _mm256_add_pd(cbufi[2], s2);
          cbufi[3] = _mm256_add_pd(cbufi[3], s3);
        }

        // write back to c
        for (size_t l = 0; l < MB; l++) {
          __m256d &cl = *(__m256d*)&(c[j+l][k]);
          cl = _mm256_add_pd(cl, cbuf0[l]);
        }
      }
    }
  }
}

