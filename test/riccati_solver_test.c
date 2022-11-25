#include "riccati/riccati_solver.h"

#include <stdio.h>
#include <stdlib.h>

#include "riccati/constants.h"
#include "simpletest/simpletest.h"
#include "slap/linalg.h"
#include "test_utils.h"

const int nstates = 3;
const int ninputs = 2;
const int nhorizon = 5;
const double Q[9] = {1, 0, 0, 0, 0.5, 0, 0, 0, 0.4};
const double R[4] = {0.1, 0, 0, 0.2};
const double H[6] = {0.1, 0, 0, 0.2, 0, 0};
const double q[3] = {-0.1, -0.2, -0.3};
const double r[2] = {0.1, 0.2};
const double c = 10.4;

const double A[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
const double B[6] = {.1, 0, 0, 0, .1, 0};
const double f[3] = {-0.1, 0.2, 0.3};

void TestNewRiccatiSolver() {
  RiccatiSolver* solver = ulqr_NewRiccatiSolver(nstates, ninputs, nhorizon);

  TEST(solver->nhorizon = nhorizon);
  TEST(solver->nstates = nstates);
  TEST(solver->ninputs = ninputs);
  TEST(solver->nvars == nhorizon * (2 * nstates + ninputs) - ninputs);
  const double tol = 1e-8;
  TESTAPPROX(solver->t_solve_ms, 0.0, tol);
  TESTAPPROX(solver->t_backward_pass_ms, 0.0, tol);
  TESTAPPROX(solver->t_forward_pass_ms, 0.0, tol);

  // Set initial state
  double x0_data[3] = {0, -1, 3.2};  // NOLINT
  ulqr_SetInitialState(solver, x0_data);
  TEST(SumOfSquaredError(solver->x0.data, x0_data, nstates) < tol);

  ulqr_FreeRiccatiSolver(&solver);
  TEST(solver == NULL);
}

void TestSetCost() {
  RiccatiSolver* solver = ulqr_NewRiccatiSolver(nstates, ninputs, nhorizon);

  int out = ulqr_SetCost(solver, Q, R, H, q, r, c, 0, 2);
  TEST(out == kOk);
  const double tol = 1e-8;
  for (int k = 0; k < 2; ++k) {
    TEST(SumOfSquaredError(ulqr_GetQ(solver, k)->data, Q, nstates * nstates) < tol);
    TEST(SumOfSquaredError(ulqr_GetR(solver, k)->data, R, ninputs * ninputs) < tol);
    TEST(SumOfSquaredError(ulqr_GetH(solver, k)->data, H, ninputs * nstates) < tol);
    TEST(SumOfSquaredError(ulqr_Getq(solver, k)->data, q, nstates) < tol);
    TEST(SumOfSquaredError(ulqr_Getr(solver, k)->data, r, ninputs) < tol);
    TESTAPPROX(ulqr_Getc(solver, k), c, tol);
  }
  for (int k = 2; k < nhorizon; ++k) {
    TEST(slap_OneNorm(ulqr_GetQ(solver, k)) < tol);
    TEST(slap_OneNorm(ulqr_GetR(solver, k)) < tol);
    TEST(slap_OneNorm(ulqr_GetH(solver, k)) < tol);
    TEST(slap_OneNorm(ulqr_Getq(solver, k)) < tol);
    TEST(slap_OneNorm(ulqr_Getr(solver, k)) < tol);
    TESTAPPROX(ulqr_Getc(solver, k), 0.0, tol);
  }

  out = ulqr_SetCost(solver, Q, R, NULL, NULL, NULL, 0, 2, nhorizon);
  TEST(out == kOk);
  for (int k = 0; k < nhorizon; ++k) {
    TEST(SumOfSquaredError(ulqr_GetQ(solver, k)->data, Q, nstates * nstates) < tol);
    TEST(SumOfSquaredError(ulqr_GetR(solver, k)->data, R, ninputs * ninputs) < tol);
  }
  for (int k = 2; k < nhorizon; ++k) {
    TEST(slap_OneNorm(ulqr_GetH(solver, k)) < tol);
    TEST(slap_OneNorm(ulqr_Getq(solver, k)) < tol);
    TEST(slap_OneNorm(ulqr_Getr(solver, k)) < tol);
    TESTAPPROX(ulqr_Getc(solver, k), 0.0, tol);
  }

  // Test bad inputs
  out = ulqr_SetCost(solver, NULL, R, NULL, NULL, NULL, 0, 2, nhorizon);
  TEST(out = kBadInput);
  out = ulqr_SetCost(solver, Q, NULL, H, q, r, 0, 2, nhorizon);
  TEST(out = kBadInput);
  out = ulqr_SetCost(NULL, Q, R, H, q, r, 0, 2, nhorizon);
  TEST(out = kBadInput);
  out = ulqr_SetCost(solver, Q, R, H, q, r, 0, -1, nhorizon);
  TEST(out = kBadInput);
  out = ulqr_SetCost(solver, Q, R, H, q, r, 0, 0, nhorizon + 1);
  TEST(out = kBadInput);

  ulqr_FreeRiccatiSolver(&solver);
}

void TestRiccatiGetters() {
  RiccatiSolver* solver = ulqr_NewRiccatiSolver(nstates, ninputs, nhorizon);

  for (int k = 0; k < nhorizon; ++k) {
    TEST(ulqr_GetA(solver, k)->rows == nstates);
    TEST(ulqr_GetA(solver, k)->cols == nstates);
    TEST(ulqr_GetB(solver, k)->rows == nstates);
    TEST(ulqr_GetB(solver, k)->cols == ninputs);
    TEST(ulqr_Getf(solver, k)->rows == nstates);
    TEST(ulqr_Getf(solver, k)->cols == 1);
    TEST(ulqr_GetFeedbackGain(solver, k)->rows == ninputs);
    TEST(ulqr_GetFeedbackGain(solver, k)->cols == nstates);
    TEST(ulqr_GetFeedforwardGain(solver, k)->rows == ninputs);
    TEST(ulqr_GetFeedforwardGain(solver, k)->cols == 1);
    TEST(ulqr_GetCostToGoHessian(solver, k)->rows == nstates);
    TEST(ulqr_GetCostToGoHessian(solver, k)->cols == nstates);
    TEST(ulqr_GetCostToGoGradient(solver, k)->rows == nstates);
    TEST(ulqr_GetCostToGoGradient(solver, k)->cols == 1);
    TEST(ulqr_GetQxx(solver, k)->rows == nstates);
    TEST(ulqr_GetQxx(solver, k)->cols == nstates);
    TEST(ulqr_GetQux(solver, k)->rows == ninputs);
    TEST(ulqr_GetQux(solver, k)->cols == nstates);
    TEST(ulqr_GetQuu(solver, k)->rows == ninputs);
    TEST(ulqr_GetQuu(solver, k)->cols == ninputs);
    TEST(ulqr_GetQx(solver, k)->rows == nstates);
    TEST(ulqr_GetQx(solver, k)->cols == 1);
    TEST(ulqr_GetQu(solver, k)->rows == ninputs);
    TEST(ulqr_GetQu(solver, k)->cols == 1);
    TEST(ulqr_GetDual(solver, k)->rows == nstates);
    TEST(ulqr_GetDual(solver, k)->cols == 1);
  }

  ulqr_FreeRiccatiSolver(&solver);
}

void TestSetDynamics() {
  RiccatiSolver* solver = ulqr_NewRiccatiSolver(nstates, ninputs, nhorizon);

  int out = ulqr_SetDynamics(solver, A, B, f, 0, 3);
  TEST(out == kOk);
  const double tol = 1e-8;
  for (int k = 0; k < 3; ++k) {
    TEST(SumOfSquaredError(ulqr_GetA(solver, k)->data, A, nstates * nstates) < tol);
    TEST(SumOfSquaredError(ulqr_GetB(solver, k)->data, B, nstates * ninputs) < tol);
    TEST(SumOfSquaredError(ulqr_Getf(solver, k)->data, f, nstates) < tol);
  }

  for (int k = 3; k < nhorizon; ++k) {
    TEST(slap_OneNorm(ulqr_GetA(solver, k)) < tol);
    TEST(slap_OneNorm(ulqr_GetB(solver, k)) < tol);
    TEST(slap_OneNorm(ulqr_Getf(solver, k)) < tol);
  }

  out = ulqr_SetDynamics(solver, A, B, NULL, 0, nhorizon);
  TEST(out == kOk);
  for (int k = 0; k < nhorizon; ++k) {
    TEST(SumOfSquaredError(ulqr_GetA(solver, k)->data, A, nstates * nstates) < tol);
    TEST(SumOfSquaredError(ulqr_GetB(solver, k)->data, B, nstates * ninputs) < tol);
    if (k >= 3) {
      TESTAPPROX(slap_OneNorm(ulqr_Getf(solver, k)), 0.0, tol);
    }
  }

  // Test bad inputs
  out = ulqr_SetDynamics(solver, NULL, NULL, f, 0, nhorizon);
  TEST(out = kBadInput);
  out = ulqr_SetDynamics(solver, A, NULL, f, 0, nhorizon);
  TEST(out = kBadInput);
  out = ulqr_SetDynamics(solver, NULL, B, f, 0, nhorizon);
  TEST(out = kBadInput);
  out = ulqr_SetDynamics(solver, A, B, f, -1, nhorizon);
  TEST(out = kBadInput);
  out = ulqr_SetDynamics(solver, A, B, f, 0, nhorizon + 1);
  TEST(out = kBadInput);

  ulqr_FreeRiccatiSolver(&solver);
}

int main() {
  // TestNewRiccatiSolver();
  // TestSetCost();
  // TestRiccatiGetters();
  TestSetDynamics();
  PrintTestResult();
  return TestResult();
}
