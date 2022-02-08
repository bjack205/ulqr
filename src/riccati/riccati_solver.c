#include "riccati/riccati_solver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constants.h"
#include "lqr_data.h"
#include "slap/matrix.h"

bool CheckBadIndex(const RiccatiSolver* solver, int k) {
  if (k < 0 || k > solver->nhorizon) {
    printf("ERROR: Invalid knot point range. Must be in interval [0,%d)\n",
           solver->nhorizon);
    return true;
  }
  return false;
}

RiccatiSolver* ulqr_NewRiccatiSolver(int nstates, int ninputs, int nhorizon) {
  int nvars = (2 * nstates + ninputs) * nhorizon - ninputs;

  int lqrdata_size = LQRDataSize(nstates, ninputs);
  int x0_size = ninputs;  // TODO (brian): avoid allocating extra data at last time step?
  int traj_size = nhorizon * (nstates + ninputs);
  int total_size = lqrdata_size * nhorizon + x0_size + traj_size;

  // Allocate all the numeric data
  double* data = (double*)malloc(total_size * sizeof(double));
  if (!data) {
    printf("ERROR: Failed to allocate memory for RiccatiSolver.\n");
    return NULL;
  }
  memset(data, 0, total_size * sizeof(double));

  // Separate into chunks
  double* lqrdata_data = data;
  double* x0_data = lqrdata_data + lqrdata_size;
  double* traj_data = x0_data + x0_size;

  // Allocate the solver
  RiccatiSolver* solver = (RiccatiSolver*)malloc(sizeof(RiccatiSolver));
  if (!solver) {
    printf("ERROR: Failed to allocate memory for RiccatiSolver.\n");
    free(data);
    return NULL;
  }

  // Allocate the trajectory
  KnotPoint* trajectory = (KnotPoint*)malloc(sizeof(KnotPoint) * nhorizon);
  if (!trajectory) {
    printf("ERROR: Failed to allocated memory for trajectory.\n");
    free(data);
    free(solver);
    return NULL;
  }
  const double h = 0.1;  // TODO (brian): pull this from an input
  for (int k = 0; k < nhorizon; ++k) {
    ulqr_InitializeKnotPoint(trajectory + k, nstates, ninputs,
                             traj_data + (nstates + ninputs) * k, h * k, h);
  }

  // Allocate space for LQRData array
  LQRData* lqrdata = (LQRData*)malloc(nhorizon * sizeof(LQRData));
  if (!lqrdata) {
    printf("ERROR: Failed to allocate LQR Data for RiccatiSolver.\n");
    free(data);
    free(solver);
    free(trajectory);
    return NULL;
  }

  // Initialize all the LQRData
  for (int k = 0; k < nhorizon; ++k) {
    int out =
        ulqr_InitializeLQRData(lqrdata + k, nstates, ninputs, data + k * lqrdata_size);
    if (out != kOk) {
      free(data);
      free(solver);
      free(lqrdata);
      free(trajectory);
      return NULL;
    }
  }

  // Initialize the solver
  solver->nhorizon = nhorizon;
  solver->nstates = nstates;
  solver->ninputs = ninputs;
  solver->nvars = nvars;
  solver->Z = trajectory;
  solver->lqrdata = lqrdata;
  solver->data = data;
  solver->x0.data = x0_data;
  slap_SetMatrixSize(&solver->x0, nstates, 1);
  solver->t_solve_ms = 0.0;
  solver->t_backward_pass_ms = 0.0;
  solver->t_forward_pass_ms = 0.0;
  return solver;
}

int ulqr_FreeRiccatiSolver(RiccatiSolver** solver_ptr) {
  RiccatiSolver* solver = *solver_ptr;
  if (!solver) {
    return -1;
  }
  free(solver->data);
  free(solver->lqrdata);
  free(solver->Z);
  free(solver);
  *solver_ptr = NULL;
  return 0;
}

enum ulqr_ReturnCode ulqr_SetInitialState(RiccatiSolver* solver, double* x0) {
  int out = kOk;
  if (solver) {
    int slap_out = slap_MatrixCopyFromArray(&solver->x0, x0);
    if (slap_out != 0) {
      out = kLinearAlgebraError;
    }
  } else {
    out = kBadInput;
  }
  return out;
}

int ulqr_PrintRiccatiSummary(RiccatiSolver* solver) {
  if (!solver) {
    return -1;
  }
  printf("NDLQR Riccati Solve Summary\n");
  double t_solve = solver->t_solve_ms;
  double t_bp = solver->t_backward_pass_ms;
  double t_fp = solver->t_forward_pass_ms;
  printf("  Solve time:    %.2f ms\n", t_solve);
  printf("  Backward Pass: %.2f ms (%.1f %% of total)\n", t_bp, t_bp / t_solve * 100.0);
  printf("  Foward Pass:   %.2f ms (%.1f %% of total)\n", t_fp, t_fp / t_solve * 100.0);
  return 0;
}

int ulqr_GetNumVars(RiccatiSolver* solver) { return solver->nvars; }

enum ulqr_ReturnCode ulqr_SetCost(RiccatiSolver* solver, const double* Q, const double* R,
                                  const double* H, const double* q, const double* r,
                                  double c, int k_start, int k_end) {
  // Check inputs
  if (!solver) {
    return kBadInput;
  }
  if (!Q || !R) {
    printf("ERROR: Both Q and R must be specified when setting the cost.\n");
    return kBadInput;
  }
  if (CheckBadIndex(solver, k_start)) {
    return kBadInput;
  }
  if (CheckBadIndex(solver, k_end)) {
    return kBadInput;
  }
  if (k_start >= k_end) {
    printf("WARNING: Specified an empty knot point interval: [%d,%d).\n", k_start, k_end);
  }

  // Copy into problem
  for (int k = k_start; k < k_end; ++k) {
    LQRData* lqrdata = solver->lqrdata + k;
    slap_MatrixCopyFromArray(&lqrdata->Q, Q);
    slap_MatrixCopyFromArray(&lqrdata->R, R);
    if (H) {
      slap_MatrixCopyFromArray(&lqrdata->H, H);
    }
    if (q) {
      slap_MatrixCopyFromArray(&lqrdata->q, q);
    }
    if (r) {
      slap_MatrixCopyFromArray(&lqrdata->r, r);
    }
    *lqrdata->c = c;
  }
  return kOk;
}

enum ulqr_ReturnCode ulqr_SetDynamics(RiccatiSolver* solver, const double* A,
                                      const double* B, const double* f, int k_start,
                                      int k_end) {
  // Check inputs
  if (!solver) {
    return kBadInput;
  }
  if (!A || !B) {
    printf("ERROR: Both A and B must be specified when setting the dynamics.\n");
    return kBadInput;
  }
  if (CheckBadIndex(solver, k_start)) {
    return kBadInput;
  }
  if (CheckBadIndex(solver, k_end)) {
    return kBadInput;
  }

  // Copy into problem
  for (int k = k_start; k < k_end; ++k) {
    int out = 0;
    LQRData* lqrdata = solver->lqrdata + k;
    out += slap_MatrixCopyFromArray(&lqrdata->A, A);
    out += slap_MatrixCopyFromArray(&lqrdata->B, B);
    if (f) {
      out += slap_MatrixCopyFromArray(&lqrdata->f, f);
    }
    if (out != 0) {
      return kLinearAlgebraError;
    }
  }
  return kOk;
}

Matrix* ulqr_GetA(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->A; }
Matrix* ulqr_GetB(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->B; }
Matrix* ulqr_Getf(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->f; }
Matrix* ulqr_GetQ(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->Q; }
Matrix* ulqr_GetR(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->R; }
Matrix* ulqr_GetH(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->H; }
Matrix* ulqr_Getq(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->q; }
Matrix* ulqr_Getr(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->r; }
double ulqr_Getc(RiccatiSolver* solver, int k) { return *(solver->lqrdata + k)->c; }

Matrix* ulqr_GetFeedbackGain(RiccatiSolver* solver, int k) {
  return &(solver->lqrdata + k)->K;
}
Matrix* ulqr_GetFeedforwardGain(RiccatiSolver* solver, int k) {
  return &(solver->lqrdata + k)->d;
}
Matrix* ulqr_GetCostToGoHessian(RiccatiSolver* solver, int k) {
  return &(solver->lqrdata + k)->P;
}
Matrix* ulqr_GetCostToGoGradient(RiccatiSolver* solver, int k) {
  return &(solver->lqrdata + k)->p;
}
Matrix* ulqr_GetQxx(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->Qxx; }
Matrix* ulqr_GetQuu(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->Quu; }
Matrix* ulqr_GetQux(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->Qux; }
Matrix* ulqr_GetQx(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->Qx; }
Matrix* ulqr_GetQu(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->Qu; }

Matrix* ulqr_GetState(RiccatiSolver* solver, int k) {
  return ulqr_GetKnotpointState(solver->Z + k);
}
Matrix* ulqr_GetInput(RiccatiSolver* solver, int k) {
  return ulqr_GetKnotpointInput(solver->Z + k);
}
Matrix* ulqr_GetDual(RiccatiSolver* solver, int k) { return &(solver->lqrdata + k)->y; }
