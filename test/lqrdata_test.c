#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lqr/lqr_data.h"
#include "lqr/lqr_problem.h"
#include "simpletest/simpletest.h"
#include "slap/matrix.h"

double A[4] = {1, 0, 1, 1};  // NOLINT
double B[2] = {1, 2};        // NOLINT
double d[2] = {4, 5};        // NOLINT
double Q[4] = {2, 0, 0, 2};  // NOLINT
double R[1] = {0.1};         // NOLINT
double H[2] = {0, 0.1};      // NOLINT
double q[2] = {0.1, 0.2};    // NOLINT
double r[1] = {-0.6};        // NOLINT
double c = 12.5;             // NOLINT

void TestNewLQRData(int nstates, int ninputs) {
  LQRData* data = ulqr_NewLQRData(nstates, ninputs);
  TEST(data->nstates == nstates);
  TEST(data->ninputs == ninputs);
  Matrix mat = ulqr_GetA(data);
  TEST(mat.rows == nstates);
  TEST(mat.cols == nstates);
  mat = ulqr_GetB(data);
  TEST(mat.rows == nstates);
  TEST(mat.cols == ninputs);
  mat = ulqr_Getd(data);
  TEST(mat.rows == nstates);
  TEST(mat.cols == 1);
  mat = ulqr_GetQ(data);
  TEST(mat.rows == nstates);
  TEST(mat.cols == nstates);
  mat = ulqr_GetR(data);
  TEST(mat.rows == ninputs);
  TEST(mat.cols == ninputs);
  mat = ulqr_GetH(data);
  TEST(mat.rows == ninputs);
  TEST(mat.cols == nstates);
  mat = ulqr_Getq(data);
  TEST(mat.rows == nstates);
  TEST(mat.cols == 1);
  mat = ulqr_Getr(data);
  TEST(mat.rows == ninputs);
  TEST(mat.cols == 1);

  ulqr_FreeLQRData(&data);
  TEST(data == NULL);
}

double SumOfSquaredError(double* x, double* y, int len) {
  double err = 0;
  for (int i = 0; i < len; ++i) {
    double diff = x[i] - y[i];
    err += diff * diff;
  }
  return sqrt(err);
}

void TestInitializeLQRData() {
  const int nstates = 2;
  const int ninputs = 1;
  LQRData* data = ulqr_NewLQRData(nstates, ninputs);
  int out = ulqr_InitializeLQRData(data, Q, R, H, q, r, c, A, B, d);
  TEST(out == 0);
  const double tol = 1e-8;
  TEST(SumOfSquaredError(Q, data->Q, nstates * nstates) < tol);
  TEST(SumOfSquaredError(R, data->R, ninputs * ninputs) < tol);
  TEST(SumOfSquaredError(H, data->H, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(q, data->q, nstates * 1) < tol);
  TEST(SumOfSquaredError(r, data->r, ninputs * 1) < tol);
  TEST(fabs(c - *data->c) < tol);
  TEST(SumOfSquaredError(A, data->A, nstates * nstates) < tol);
  TEST(SumOfSquaredError(B, data->B, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(d, data->d, nstates * 1) < tol);

  // Test passing in bad pointer
  out = ulqr_InitializeLQRData(data, NULL, R, H, q, r, c, A, B, d);
  TEST(out != 0);

  out = ulqr_InitializeLQRData(data, Q, R, H, q, r, c, A, B, NULL);
  TEST(out != 0);

  // Free then try to initialize again
  ulqr_FreeLQRData(&data);
  TEST(data == NULL);
  out = ulqr_InitializeLQRData(data, Q, R, H, q, r, c, A, B, d);
  TEST(out != 0);
}

void TestLQRDataCopy() {
  const int nstates = 2;
  const int ninputs = 1;
  LQRData* data0 = ulqr_NewLQRData(nstates, ninputs);
  LQRData* data1 = ulqr_NewLQRData(nstates, ninputs);
  ulqr_InitializeLQRData(data0, Q, R, H, q, r, c, A, B, d);
  ulqr_CopyLQRData(data1, data0);

  const double tol = 1e-8;
  TEST(SumOfSquaredError(Q, data1->Q, nstates * nstates) < tol);
  TEST(SumOfSquaredError(R, data1->R, ninputs * ninputs) < tol);
  TEST(SumOfSquaredError(H, data1->H, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(q, data1->q, nstates * 1) < tol);
  TEST(SumOfSquaredError(r, data1->r, ninputs * 1) < tol);
  TEST(fabs(c - *data1->c) < tol);
  TEST(SumOfSquaredError(A, data1->A, nstates * nstates) < tol);
  TEST(SumOfSquaredError(B, data1->B, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(d, data1->d, nstates * 1) < tol);
  ulqr_FreeLQRData(&data0);
  ulqr_FreeLQRData(&data1);
}

void TestPrintLQRData() {
  const int nstates = 2;
  const int ninputs = 1;
  LQRData* data = ulqr_NewLQRData(nstates, ninputs);
  ulqr_PrintLQRData(data);
  ulqr_FreeLQRData(&data);
}

int main() {
  int nstates = 6;  // NOLINT
  int ninputs = 3;  // NOLINT
  TestNewLQRData(nstates, ninputs);
  TestInitializeLQRData();
  TestLQRDataCopy();
  TestPrintLQRData();
  PrintTestResult();
  return TestResult();
}
