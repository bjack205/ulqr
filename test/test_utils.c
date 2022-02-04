#include "test_utils.h"

#include <math.h>

double SumOfSquaredError(const double* x, const double* y, int len) {
  double err = 0;
  for (int i = 0; i < len; ++i) {
    double diff = x[i] - y[i];
    err += diff * diff;
  }
  return sqrt(err);
}
