#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "riccati/constants.h"
#include "riccati/lqr_data.h"
#include "simpletest/simpletest.h"
#include "slap/matrix.h"
#include "test_utils.h"

double A[4] = {1, 0, 1, 1};  // NOLINT
double B[2] = {1, 2};        // NOLINT
double f[2] = {4, 5};        // NOLINT
double Q[4] = {2, 0, 0, 2};  // NOLINT
double R[1] = {0.1};         // NOLINT
double H[2] = {0, 0.1};      // NOLINT
double q[2] = {0.1, 0.2};    // NOLINT
double r[1] = {-0.6};        // NOLINT
double c = 12.5;             // NOLINT

void SetLQRData(LQRData* lqrdata) {
  slap_MatrixCopyFromArray(&lqrdata->Q, Q);
  slap_MatrixCopyFromArray(&lqrdata->R, R);
  slap_MatrixCopyFromArray(&lqrdata->H, H);
  slap_MatrixCopyFromArray(&lqrdata->q, q);
  slap_MatrixCopyFromArray(&lqrdata->r, r);
  slap_MatrixCopyFromArray(&lqrdata->A, A);
  slap_MatrixCopyFromArray(&lqrdata->B, B);
  slap_MatrixCopyFromArray(&lqrdata->f, f);
  *lqrdata->c = c;
}

void TestInitializeLQRData() {
  const int nstates = 2;
  const int ninputs = 1;
  LQRData* lqrdata = (LQRData*)malloc(sizeof(LQRData));
  double* data = (double*)calloc(LQRDataSize(nstates, ninputs), sizeof(double));
  ulqr_InitializeLQRData(lqrdata, nstates, ninputs, data);

  // Check sizes
  TEST(lqrdata->nstates == nstates);
  TEST(lqrdata->ninputs == ninputs);
  Matrix mat = lqrdata->A;
  TEST(mat.rows == nstates);
  TEST(mat.cols == nstates);
  mat = lqrdata->B;
  TEST(mat.rows == nstates);
  TEST(mat.cols == ninputs);
  mat = lqrdata->f;
  TEST(mat.rows == nstates);
  TEST(mat.cols == 1);
  mat = lqrdata->Q;
  TEST(mat.rows == nstates);
  TEST(mat.cols == nstates);
  mat = lqrdata->R;
  TEST(mat.rows == ninputs);
  TEST(mat.cols == ninputs);
  mat = lqrdata->H;
  TEST(mat.rows == ninputs);
  TEST(mat.cols == nstates);
  mat = lqrdata->q;
  TEST(mat.rows == nstates);
  TEST(mat.cols == 1);
  mat = lqrdata->r;
  TEST(mat.rows == ninputs);
  TEST(mat.cols == 1);

  TEST(lqrdata->K.rows == ninputs);
  TEST(lqrdata->K.cols == nstates);
  TEST(lqrdata->d.rows == ninputs);
  TEST(lqrdata->d.cols == 1);

  TEST(lqrdata->P.rows == nstates);
  TEST(lqrdata->P.cols == nstates);
  TEST(lqrdata->p.rows == nstates);
  TEST(lqrdata->p.cols == 1);

  TEST(lqrdata->Qxx.rows == nstates);
  TEST(lqrdata->Qxx.cols == nstates);
  TEST(lqrdata->Qux.rows == ninputs);
  TEST(lqrdata->Qux.cols == nstates);
  TEST(lqrdata->Quu.rows == ninputs);
  TEST(lqrdata->Quu.cols == ninputs);

  TEST(lqrdata->y.rows == nstates);
  TEST(lqrdata->y.cols == 1);
  TEST((lqrdata->y.data + nstates) - (lqrdata->Q.data) + 1 == LQRDataSize(nstates, ninputs));

  const double tol = 1e-8;
  SetLQRData(lqrdata);
  TEST(SumOfSquaredError(Q, lqrdata->Q.data, nstates * nstates) < tol);
  TEST(SumOfSquaredError(R, lqrdata->R.data, ninputs * ninputs) < tol);
  TEST(SumOfSquaredError(H, lqrdata->H.data, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(q, lqrdata->q.data, nstates * 1) < tol);
  TEST(SumOfSquaredError(r, lqrdata->r.data, ninputs * 1) < tol);
  TEST(fabs(c - *lqrdata->c) < tol);
  TEST(SumOfSquaredError(A, lqrdata->A.data, nstates * nstates) < tol);
  TEST(SumOfSquaredError(B, lqrdata->B.data, nstates * ninputs) < tol);
  TEST(SumOfSquaredError(f, lqrdata->f.data, nstates * 1) < tol);

  // Test passing in bad pointer
  int out = ulqr_InitializeLQRData(NULL, nstates, ninputs, data);
  TEST(out == kBadInput);

  out = ulqr_InitializeLQRData(lqrdata, -1, ninputs, data);
  TEST(out == kBadInput);

  out = ulqr_InitializeLQRData(lqrdata, nstates, 0, data);
  TEST(out == kBadInput);

  out = ulqr_InitializeLQRData(lqrdata, nstates, ninputs, NULL);
  TEST(out == kBadInput);

  free(lqrdata);
  free(data);
}

void TestLQRDataCopy() {
  const int nstates = 2;
  const int ninputs = 1;
  LQRData* lqrdata = (LQRData*)malloc(2 * sizeof(LQRData));
  int datasize = LQRDataSize(nstates, ninputs);
  double* data = (double*)malloc(2 * datasize * sizeof(double));
  ulqr_InitializeLQRData(lqrdata + 0, nstates, ninputs, data);
  ulqr_InitializeLQRData(lqrdata + 1, nstates, ninputs, data + datasize);

  LQRData* data0 = lqrdata + 0;
  LQRData* data1 = lqrdata + 1;
  SetLQRData(data0);
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
  TEST(SumOfSquaredError(f, data1->f.data, nstates * 1) < tol);

  free(lqrdata);
  free(data);
}

void InitLQRData(LQRData* lqrdata, double* data) { lqrdata->c = data; }

void TestLQRDataArray() {
  const int nhorizon = 3;
  const int nstates = 3;
  const int ninputs = 2;
  int lqrdata_size = LQRDataSize(nstates, ninputs);
  LQRData* lqrdata = (LQRData*)malloc(nhorizon * sizeof(LQRData));
  double* data = (double*)calloc(lqrdata_size * nhorizon, sizeof(double));
  for (int k = 0; k < nhorizon; ++k) {
    ulqr_InitializeLQRData(lqrdata + k, nstates, ninputs, data + k * lqrdata_size);
  }

  double c = 0.0;
  for (int k = 0; k < 2; ++k) {
    c += *(lqrdata + k)->c;
  }
  TEST(c == 0);

  free(lqrdata);
  free(data);
}

int main() {
  // TestInitializeLQRData();
  // TestLQRDataCopy();
  TestLQRDataArray();
  PrintTestResult();
  return TestResult();
}
