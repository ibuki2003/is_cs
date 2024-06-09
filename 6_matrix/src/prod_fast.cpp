#include "prod_fast.hpp"
#include <cassert>
#include <config.hpp>
#include <cstring>
#include <immintrin.h>

constexpr size_t NB3 = 256;
constexpr size_t MB3 = 256;
constexpr size_t KB3 = 256;

constexpr size_t NB2 = 64;
constexpr size_t MB2 = 128;
constexpr size_t KB2 = 256;

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

  assert(n % NB3 == 0);
  assert(n % MB3 == 0);
  assert(n % KB3 == 0);

  memset(c.data, 0, sizeof(double) * n * n);

  __align_avx__ double abuf2[MB2/4][KB2][4];
  __align_avx__ double bbuf2[NB2/4][KB2][4];
  __align_avx__ __m256d cbuf2[MB2][NB2/4];

  // LOOP 3
  for (size_t i3 = 0; i3 < n; i3 += NB3) {
    for (size_t j3 = 0; j3 < n; j3 += MB3) {
      for (size_t k3 = 0; k3 < n; k3 += KB3) {

        // LOOP 2
        for (size_t k2 = 0; k2 < KB3; k2 += KB2) {
          for (size_t j2 = 0; j2 < MB3; j2 += MB2) {

            // copy A[j][k];
            for (size_t x = 0; x < MB2; ++x) {
              for (size_t y = 0; y < KB2; ++y) {
                abuf2[x/4][y][x%4] = a[j3 + j2 + x][k3 + k2 + y];
              }
            }

            for (size_t i2 = 0; i2 < NB3; i2 += NB2) {

              memset(cbuf2, 0, sizeof(cbuf2));

              // copy B[k][i]
              for (size_t x = 0; x < KB2; ++x) {
                for (size_t y = 0; y < NB2; y += 4) {
                  bbuf2[y/4][x][0] = b[k3 + k2 + x][i3 + i2 + y + 0];
                  bbuf2[y/4][x][1] = b[k3 + k2 + x][i3 + i2 + y + 1];
                  bbuf2[y/4][x][2] = b[k3 + k2 + x][i3 + i2 + y + 2];
                  bbuf2[y/4][x][3] = b[k3 + k2 + x][i3 + i2 + y + 3];
                }
              }

              // LOOP 1
              for (size_t i1 = 0; i1 < NB2; i1 += 4) {
                for (size_t j1 = 0; j1 < MB2; j1 += 4) {
                  __m256d s0 = _mm256_setzero_pd();
                  __m256d s1 = _mm256_setzero_pd();
                  __m256d s2 = _mm256_setzero_pd();
                  __m256d s3 = _mm256_setzero_pd();

                  for (size_t k1 = 0; k1 < KB2; k1++) {
                    prod44(
                      _mm256_load_pd(abuf2[j1/4][k1]),
                      _mm256_load_pd(bbuf2[i1/4][k1]),
                      s0, s1, s2, s3
                    );
                  }
                  shuf44(s0, s1, s2, s3);
                  /* _mm256_store_pd(&c[j3 + j2 + j1 + 0][i3 + i2 + i1], s0); */
                  /* _mm256_store_pd(&c[j3 + j2 + j1 + 1][i3 + i2 + i1], s1); */
                  /* _mm256_store_pd(&c[j3 + j2 + j1 + 2][i3 + i2 + i1], s2); */
                  /* _mm256_store_pd(&c[j3 + j2 + j1 + 3][i3 + i2 + i1], s3); */

                  cbuf2[j1 + 0][i1/4] = s0;
                  cbuf2[j1 + 1][i1/4] = s1;
                  cbuf2[j1 + 2][i1/4] = s2;
                  cbuf2[j1 + 3][i1/4] = s3;
                }
              }

              // cbuf to c
              for (size_t x = 0; x < MB2; ++x) {
                for (size_t y = 0; y < NB2; y += 4) {
                  auto& cc = c[j3 + j2 + x][i3 + i2 + y];
                  _mm256_store_pd(
                    &cc,
                    _mm256_add_pd(_mm256_load_pd(&cc), cbuf2[x][y/4])
                  );
                }
              }

            }
          }
        }

      }
    }
  }

}

