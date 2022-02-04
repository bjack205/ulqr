#include "simpletest/simpletest.h"
#include "riccati/riccati_solver.h"

void TestNewRiccatiSolver() {
  const int nstates = 3;
  const int ninputs = 2;
  const int nhorizon = 5;
  RiccatiSolver* solver = ulqr_NewRiccatiSolver(nstates, ninputs, nhorizon);

  TEST(solver->nhorizon = nhorizon);
  TEST(solver->nstates = nstates);
  TEST(solver->ninputs = ninputs);

  ulqr_FreeRiccatiSolver(&solver);
}

int main() {
  TestNewRiccatiSolver();
  PrintTestResult();
  return TestResult();
}
