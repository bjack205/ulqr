#include <stdio.h>

#include "riccati/constants.h"
#include "riccati/riccati_solver.h"
#include "simpletest/simpletest.h"
#include "slap/matrix.h"
#include "slap/linalg.h"
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

void TestDoubleIntegratorCost() {  
  const double tol = 1e-10;
  const double dim = 1;
  const double h = 0.2;
  int nstates = 2 * dim;
  int ninputs = dim;
  const int nhorizon = 11;

  // Define cost weights
  Matrix Q = slap_NewMatrixZeros(nstates, nstates);
  Matrix R = slap_NewMatrixZeros(nstates, nstates);
  Matrix Qf = slap_NewMatrixZeros(nstates, nstates);

  slap_AddDiagonal(&Q, 1.0 * h);
  slap_AddDiagonal(&R, 0.1 * h);  // NOLINT
  slap_AddDiagonal(&Qf, 1.0);

  // Initial and final position
  Matrix xf = slap_NewMatrixZeros(nstates, 1);
  Matrix x0 = slap_NewMatrixZeros(nstates, 1);
  for (int i = 0; i < dim; ++i) {
    xf.data[i] = 1.0;
  }

  // Calculate cost terms
  Matrix q = slap_NewMatrixZeros(nstates, 1);
  Matrix qf = slap_NewMatrixZeros(nstates, 1);
  slap_MatrixMultiply(&Q, &xf, &q, 0, 0, -1.0, 0.0);  // q = -Q*xf
  slap_MatrixMultiply(&Qf, &xf, &qf, 0, 0, -1.0, 0.0);
  double c = 0.0;
  double cf = 0.0;
  for (int i = 0; i < nstates; ++i) {
    c += xf.data[i] * *slap_MatrixGetElement(&Q, i, i) * xf.data[i];
    cf += xf.data[i] * *slap_MatrixGetElement(&Qf, i, i) * xf.data[i];
  }
  c /= 2;
  cf /= 2;

  // Create the solver
  RiccatiSolver* solver = ulqr_NewRiccatiSolver(nstates, ninputs, nhorizon);

  // Set trajectory to zeros
  for (int k = 0; k < nhorizon; ++k) {
    slap_MatrixSetConst(ulqr_GetState(solver, k), 0.0);
    if (k < nhorizon - 1) {
      slap_MatrixSetConst(ulqr_GetInput(solver, k), 0.0);
    }
  }

  // Initialize the cost
  enum ulqr_ReturnCode out;
  out = ulqr_SetCost(solver, Q.data, R.data, NULL, q.data, NULL, c, 0, nhorizon - 1);
  TEST(out == kOk);
  out = ulqr_SetCost(solver, Qf.data, R.data, NULL, qf.data, NULL, cf, nhorizon - 1, nhorizon);
  TEST(out == kOk);

  // Calculate the cost
  double cost = ulqr_CalcCost(solver);
  const double cost_ans = 1.5;
  TESTAPPROX(cost, cost_ans, tol);
  printf("Cost = %f\n", cost);

  // Set trajectory to some values and get the cost again
  for (int k = 0; k < nhorizon; ++k) {
    const double xval = 1.2;
    const double uval = 0.5;
    slap_MatrixSetConst(ulqr_GetState(solver, k), xval);
    if (k < nhorizon - 1) {
      slap_MatrixSetConst(ulqr_GetInput(solver, k), uval);
    }
  }
  const double cost_ans2 = 2.245;
  cost = ulqr_CalcCost(solver);
  TESTAPPROX(cost, cost_ans2, tol);
  printf("Cost = %f\n", cost);
  
  slap_FreeMatrix(&Q);
  slap_FreeMatrix(&R);
  slap_FreeMatrix(&Qf);
  slap_FreeMatrix(&x0);
  slap_FreeMatrix(&xf);
  slap_FreeMatrix(&q);
  slap_FreeMatrix(&qf);
  ulqr_FreeRiccatiSolver(&solver);
}

int main() {
  // PrintDI(2);
  // TestDoubleIntegratorDynamics();
  TestDoubleIntegratorCost();
  PrintTestResult();
  return TestResult();
}
