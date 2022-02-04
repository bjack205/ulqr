#include "riccati/knotpoint.h"

#include <math.h>
#include <stdlib.h>

#include "riccati/constants.h"
#include "simpletest/simpletest.h"
#include "test_utils.h"

void TestInitializeKnotPoint() {
  KnotPoint z;
  const int nstates = 3;
  const int ninputs = 2;
  const double t = 1.2;
  const double h = 0.1;
  double zdata[5] = {1, 2, 3, 4, 5};  // NOLINT
  int out = ulqr_InitializeKnotPoint(&z, nstates, ninputs, zdata, t, h);
  TEST(out == kOk);
  const double tol = 1e-8;
  TESTAPPROX(ulqr_GetTime(&z), t, tol);
  TESTAPPROX(ulqr_GetTimestep(&z), h, tol);
  Matrix* x = ulqr_GetKnotpointState(&z);
  Matrix* u = ulqr_GetKnotpointInput(&z);
  TEST(x->rows == nstates);
  TEST(x->cols == 1);
  TEST(u->rows == ninputs);
  TEST(u->cols == 1);
  TESTAPPROX(x->data[0], zdata[0], tol);
  TESTAPPROX(x->data[1], zdata[1], tol);
  TESTAPPROX(x->data[2], zdata[2], tol);
  TESTAPPROX(u->data[0], zdata[3], tol);
  TESTAPPROX(u->data[1], zdata[4], tol);

  // Try bad inputs
  out = ulqr_InitializeKnotPoint(NULL, nstates, ninputs, zdata, t, h);
  TEST(out == kBadInput);

  ulqr_InitializeKnotPoint(&z, 0, ninputs, zdata, t, h);
  TEST(out == kBadInput);

  ulqr_InitializeKnotPoint(&z, nstates, ninputs, NULL, t, h);
  TEST(out == kBadInput);

  ulqr_InitializeKnotPoint(&z, nstates, ninputs, zdata, -t, h);
  TEST(out == kBadInput);

  ulqr_InitializeKnotPoint(&z, nstates, ninputs, zdata, t, -h);
  TEST(out == kBadInput);
}

int main() {
  TestInitializeKnotPoint();
  PrintTestResult();
  return TestResult();
}
