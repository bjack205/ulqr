#include <stdio.h>

#include "simpletest/simpletest.h"
#include "slap/matrix.h"
#include "test_utils.h"

void PrintDI(int dim) {
  Matrix A = slap_NewMatrix(2 * dim, 2 * dim);
  Matrix B = slap_NewMatrix(2 * dim, 1 * dim);

  const double h = 0.1;
  DiscreteDoubleIntegratorDynamics(h, dim, &A, &B);
  printf("A Matrix: \n");
  slap_PrintMatrix(&A);
  printf("\nB Matrix: \n");
  slap_PrintMatrix(&B);

  slap_FreeMatrix(&A);
  slap_FreeMatrix(&B);
}

void TestDoubleIntegratorDynamics() {
  const int dim = 2;
  const double h = 0.1;
  const double b = 0.005;

  Matrix A = slap_NewMatrix(2 * dim, 2 * dim);
  Matrix B = slap_NewMatrix(2 * dim, 1 * dim);
  // clang-format off
  // Need to enter data by column (i.e. transposed)
  double Atrue[16] = {  // NOLINT
    1, 0, 0, 0, 
    0, 1, 0, 0,
    h, 0, 1, 0,
    0, h, 0, 1
  };
  double Btrue[8] = {  // NOLINT
    b, 0, h, 0,
    0, b, 0, h
  };
  // clang-format on

  DiscreteDoubleIntegratorDynamics(h, dim, &A, &B);
  const double tol = 1e-8;
  TEST(SumOfSquaredError(A.data, Atrue, 4 * dim * dim) < tol);
  TEST(SumOfSquaredError(B.data, Btrue, 2 * dim * dim) < tol);

  slap_FreeMatrix(&A);
  slap_FreeMatrix(&B);
}

int main() {
  PrintDI(2);
  TestDoubleIntegratorDynamics();
  PrintTestResult();
  return TestResult();
}
