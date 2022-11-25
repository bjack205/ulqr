#include "riccati_solve.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lqr_data.h"
#include "riccati/riccati_solver.h"
#include "slap/matrix.h"

int ulqr_SolveRiccati(RiccatiSolver* solver) {
  if (!solver) {
    return -1;
  }
  clock_t t_start_total = clock();

  ulqr_BackwardPass(solver);
  clock_t t_start_fp = clock();
  ulqr_ForwardPass(solver);

  // Calculate timing
  const double milliseconds2seconds = 1000.0;
  clock_t t_stop = clock();
  clock_t diff_bp = t_start_fp - t_start_total;
  clock_t diff_fp = t_stop - t_start_fp;
  clock_t diff_total = t_stop - t_start_total;
  solver->t_solve_ms = diff_total * milliseconds2seconds / (double)CLOCKS_PER_SEC;
  solver->t_backward_pass_ms = diff_bp * milliseconds2seconds / (double)CLOCKS_PER_SEC;
  solver->t_forward_pass_ms = diff_fp * milliseconds2seconds / (double)CLOCKS_PER_SEC;
  return 0;
}

int ulqr_BackwardPass(RiccatiSolver* solver) {
  int nhorizon = solver->nhorizon;

  int k = nhorizon - 1;
  Matrix* Q = ulqr_GetQ(solver, k);
  Matrix* q = ulqr_Getq(solver, k);
  Matrix* Pn = ulqr_GetCostToGoHessian(solver, k);
  Matrix* pn = ulqr_GetCostToGoGradient(solver, k);
  slap_MatrixCopy(Pn, Q);
  slap_MatrixCopy(pn, q);

  // Create a matrix that treats both gains as one matrix to save an extra Cholesky solve
  // This works as long as their data is adjacent in memory
  Matrix Kd = {solver->ninputs, solver->nstates + 1, NULL};

  for (--k; k >= 0; --k) {
    Pn = ulqr_GetCostToGoHessian(solver, k);
    pn = ulqr_GetCostToGoGradient(solver, k);

    Matrix* A = ulqr_GetA(solver, k);
    Matrix* B = ulqr_GetB(solver, k);
    Matrix* f = ulqr_Getf(solver, k);
    Matrix* Q = ulqr_GetQ(solver, k);
    Matrix* q = ulqr_Getq(solver, k);
    Matrix* R = ulqr_GetR(solver, k);
    Matrix* r = ulqr_Getr(solver, k);

    // Calculate gradient terms
    Matrix* Qx = ulqr_GetQx(solver, k);
    Matrix* Qu = ulqr_GetQu(solver, k);
    Matrix* Qx_tmp = Qx + 1;
    Matrix* Qu_tmp = Qu + 1;
    slap_MatrixCopy(Qx_tmp, pn);                         // Qx = p
    slap_MatrixMultiply(Pn, f, Qx_tmp, 0, 0, 1.0, 1.0);  // Qx = P * f + p

    slap_MatrixMultiply(B, Qx_tmp, Qu, 1, 0, 1.0, 0.0);  // Qu = B' * (P * f + p)
    slap_MatrixMultiply(A, Qx_tmp, Qx, 1, 0, 1.0, 0.0);  // Qx = A' * (P * f + p)
    slap_MatrixAddition(r, Qu, 1.0);                     // Qu = r + B' * (P * f + p)
    slap_MatrixAddition(q, Qx, 1.0);                     // Qx = q + A' * (P * f + p)

    // Calculate Hessian terms
    Matrix* Qxx = ulqr_GetQxx(solver, k);
    Matrix* Qux = ulqr_GetQux(solver, k);
    Matrix* Quu = ulqr_GetQuu(solver, k);
    Matrix* Qxx_tmp = ulqr_GetQxx(solver, k + 1);
    Matrix* Qux_tmp = ulqr_GetQux(solver, k + 1);
    Matrix* Quu_tmp = ulqr_GetQuu(solver, k + 1);

    slap_MatrixCopy(Qxx, Q);
    slap_MatrixCopy(Quu, R);

    slap_MatrixMultiply(A, Pn, Qxx_tmp, 1, 0, 1.0, 0.0);   // Qxx = A'P
    slap_MatrixMultiply(B, Pn, Qux_tmp, 1, 0, 1.0, 0.0);   // Qux = B'P
    slap_MatrixMultiply(Qxx_tmp, A, Qxx, 0, 0, 1.0, 1.0);  // Qxx = Q + A'P*A
    slap_MatrixMultiply(Qux_tmp, B, Quu, 0, 0, 1.0, 1.0);  // Quu = R + B'P*B
    slap_MatrixMultiply(Qux_tmp, A, Qux, 0, 0, 1.0, 0.0);  // Qux = B'P*A

    // Calculate Gains
    Matrix* K = ulqr_GetFeedbackGain(solver, k);
    Matrix* d = ulqr_GetFeedforwardGain(solver, k);
    slap_MatrixCopy(Quu_tmp, Quu);
    slap_MatrixCopy(K, Qux);
    slap_MatrixCopy(d, Qu);

    int info = slap_CholeskyFactorize(Quu_tmp);
    if (info == slap_kCholeskyFail) {
      // TODO (sam): handle regularization
    }
    Kd.data = K->data;
    slap_CholeskySolve(Quu_tmp, &Kd);
    slap_MatrixScaleByConst(K, -1);
    slap_MatrixScaleByConst(d, -1);

    // Calulate Cost-to-Go
    Matrix* P = ulqr_GetCostToGoHessian(solver, k);
    Matrix* p = ulqr_GetCostToGoGradient(solver, k);

    slap_MatrixCopy(P, Qxx);
    slap_MatrixMultiply(Quu, K, Qux_tmp, 0, 0, 1.0, 0.0);  // Qux_tmp = Quu * K
    slap_MatrixMultiply(K, Qux_tmp, P, 1, 0, 1.0, 1.0);    // P = Qxx + K'Quu*K
    slap_MatrixMultiply(K, Qux, P, 1, 0, 1.0, 1.0);        // P = Quu + K'Quu*K + K'Qux
    slap_MatrixMultiply(Qux, K, P, 1, 0, 1.0, 1.0);        // P = Quu + K'Quu*K + K'Qux + Qux'K

    slap_MatrixCopy(p, Qx);
    slap_MatrixMultiply(Quu, d, Qu_tmp, 0, 0, 1.0, 0.0);  // Qu_tmp = Quu * d
    slap_MatrixMultiply(K, Qu_tmp, p, 1, 0, 1.0, 1.0);    // p = Qx + K'Quu*d
    slap_MatrixMultiply(K, Qu, p, 1, 0, 1.0, 1.0);        // p = Qx + K'Quu*d + K'Qu
    slap_MatrixMultiply(Qux, d, p, 1, 0, 1.0, 1.0);       // p = Qx + K'Quu*d + K'Qu + Qux'd
  }
  return 0;
}

int ulqr_ForwardPass(RiccatiSolver* solver) {
  if (!solver) {
    return -1;
  }
  int nhorizon = solver->nhorizon;
  // int nstates = solver->nstates;

  slap_MatrixCopy(ulqr_GetState(solver, 0), &solver->x0);
  int k;
  for (k = 0; k < nhorizon - 1; ++k) {
    Matrix* A = ulqr_GetA(solver, k);
    Matrix* B = ulqr_GetB(solver, k);
    Matrix* f = ulqr_Getf(solver, k);
    Matrix* Pk = ulqr_GetCostToGoHessian(solver, k);
    Matrix* pk = ulqr_GetCostToGoGradient(solver, k);
    Matrix* Kk = ulqr_GetFeedbackGain(solver, k);
    Matrix* dk = ulqr_GetFeedforwardGain(solver, k);
    Matrix* xk = ulqr_GetState(solver, k);
    Matrix* uk = ulqr_GetInput(solver, k);
    Matrix* yk = ulqr_GetDual(solver, k);
    Matrix* xn = ulqr_GetState(solver, k + 1);

    slap_MatrixCopy(yk, pk);
    slap_MatrixMultiply(Pk, xk, yk, 0, 0, 1.0, 1.0);  // y = P * x + p
    slap_MatrixCopy(uk, dk);
    slap_MatrixMultiply(Kk, xk, uk, 0, 0, 1.0, 1.0);  // un = K * x + d
    slap_MatrixCopy(xn, f);
    slap_MatrixMultiply(A, xk, xn, 0, 0, 1.0, 1.0);  // xn = A * x + f
    slap_MatrixMultiply(B, uk, xn, 0, 0, 1.0, 1.0);  // xn = A * x + B * u + f
  }
  Matrix* Pk = ulqr_GetCostToGoHessian(solver, k);
  Matrix* pk = ulqr_GetCostToGoGradient(solver, k);
  Matrix* xk = ulqr_GetState(solver, k);
  Matrix* yk = ulqr_GetDual(solver, k);
  slap_MatrixCopy(yk, pk);
  slap_MatrixMultiply(Pk, xk, yk, 0, 0, 1.0, 1.0);  // y = P * x + p
  return 0;
}
