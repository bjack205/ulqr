#include "riccati/riccati_solver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constants.h"
#include "lqr_data.h"
#include "slap/matrix.h"

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
