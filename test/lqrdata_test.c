#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "riccati/lqr_data.h"
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
  LQRData* data = ulqr_NewLQRData(nstates, ninputs, NULL);
  TEST(data->nstates == nstates);
  TEST(data->ninputs == ninputs);
  Matrix mat = data->A; 
  TEST(mat.rows == nstates);
  TEST(mat.cols == nstates);
  mat = data->B; 
  TEST(mat.rows == nstates);
  TEST(mat.cols == ninputs);
  mat = data->f; 
  TEST(mat.rows == nstates);
  TEST(mat.cols == 1);
  mat = data->Q; 
  TEST(mat.rows == nstates);
  TEST(mat.cols == nstates);
  mat = data->R; 
  TEST(mat.rows == ninputs);
  TEST(mat.cols == ninputs);
  mat = data->H; 
  TEST(mat.rows == ninputs);
  TEST(mat.cols == nstates);
  mat = data->q; 
  TEST(mat.rows == nstates);
  TEST(mat.cols == 1);
  mat = data->r; 
  TEST(mat.rows == ninputs);
  TEST(mat.cols == 1);

  TEST(data->K.rows == ninputs);
  TEST(data->K.cols == nstates);
  TEST(data->d.rows == ninputs);
  TEST(data->d.cols == 1);

  TEST(data->P.rows == nstates);
  TEST(data->P.cols == nstates);
  TEST(data->p.rows == nstates);
  TEST(data->p.cols == 1);

  TEST(data->Qux.rows == ninputs);
  TEST(data->Qux.cols == nstates);

  TEST(data->x.rows == nstates);
  TEST(data->x.cols == 1);
  TEST(data->u.rows == ninputs);
  TEST(data->u.cols == 1);
  TEST(data->y.rows == nstates);
  TEST(data->y.cols == 1);

  TEST(data->_isowner == true);

  int datasize = data->datasize; 
  ulqr_FreeLQRData(&data);
  TEST(data == NULL);

  // Test assigning into existing memory pointer
  const int offset = 10;
  double* newdata = (double*)malloc((datasize + offset) * sizeof(double));
  LQRData* lqrdata = ulqr_NewLQRData(nstates, ninputs, newdata + offset);
  TEST(lqrdata->R.data == newdata + 10 + nstates * nstates);
  TEST(lqrdata->_isowner == false);
  ulqr_FreeLQRData(&lqrdata);
  TEST(newdata != NULL);
  free(newdata);
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
  LQRData* data = ulqr_NewLQRData(nstates, ninputs, NULL);
  int out = ulqr_InitializeLQRData(data, Q, R, H, q, r, c, A, B, d);
  TEST(out == 0);
  const double tol = 1e-8;
  TEST(SumOfSquaredError(Q, data->Q.data, nstates * nstates) < tol);
  TEST(SumOfSquaredError(R, data->R.data, ninputs * ninputs) < tol);
  TEST(SumOfSquaredError(H, data->H.data, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(q, data->q.data, nstates * 1) < tol);
  TEST(SumOfSquaredError(r, data->r.data, ninputs * 1) < tol);
  TEST(fabs(c - *data->c) < tol);
  TEST(SumOfSquaredError(A, data->A.data, nstates * nstates) < tol);
  TEST(SumOfSquaredError(B, data->B.data, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(d, data->d.data, nstates * 1) < tol);

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
  LQRData* data0 = ulqr_NewLQRData(nstates, ninputs, NULL);
  LQRData* data1 = ulqr_NewLQRData(nstates, ninputs, NULL);
  ulqr_InitializeLQRData(data0, Q, R, H, q, r, c, A, B, d);
  ulqr_CopyLQRData(data1, data0);

  const double tol = 1e-8;
  TEST(SumOfSquaredError(Q, data1->Q.data, nstates * nstates) < tol);
  TEST(SumOfSquaredError(R, data1->R.data, ninputs * ninputs) < tol);
  TEST(SumOfSquaredError(H, data1->H.data, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(q, data1->q.data, nstates * 1) < tol);
  TEST(SumOfSquaredError(r, data1->r.data, ninputs * 1) < tol);
  TEST(fabs(c - *data1->c) < tol);
  TEST(SumOfSquaredError(A, data1->A.data, nstates * nstates) < tol);
  TEST(SumOfSquaredError(B, data1->B.data, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(d, data1->d.data, nstates * 1) < tol);
  ulqr_FreeLQRData(&data0);
  ulqr_FreeLQRData(&data1);
}

int main() {
  int nstates = 6;  // NOLINT
  int ninputs = 3;  // NOLINT
  TestNewLQRData(nstates, ninputs);
  TestInitializeLQRData();
  TestLQRDataCopy();
  PrintTestResult();
  return TestResult();
}
