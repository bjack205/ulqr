#include "knotpoint.h"

#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "slap/matrix.h"

enum ulqr_ReturnCode ulqr_InitializeKnotPoint(KnotPoint* z, int nstates, int ninputs,
                                              double* data, double t, double h) {
  // Input validation
  if (!z) {
    printf("ERROR: Can't pass null point to knot point when initializing.\n");
    return kBadInput;
  }
  if (nstates < 1 || ninputs < 1) {
    printf("ERROR: nstates and ninputs must be positive integers.\n");
    return kBadInput;
  }
  if (!data) {
    printf("ERROR: Cannot pass null pointer when initializing a KnotPoint.\n");
    return kBadInput;
  }
  if (t < 0 || h < 0) {
    printf("ERROR: Time and time step can't be negative.\n");
    return kBadInput;
  }

  z->x.data = data;
  slap_SetMatrixSize(&z->x, nstates, 1);
  z->u.data = data + nstates;
  slap_SetMatrixSize(&z->u, ninputs, 1);
  z->t = t;
  z->h = h;

  return kOk;
}

Matrix* ulqr_GetKnotpointState(KnotPoint* z) { return &z->x; }
Matrix* ulqr_GetKnotpointInput(KnotPoint* z) { return &z->u; }
double ulqr_GetTime(KnotPoint* z) { return z->t; }
double ulqr_GetTimestep(KnotPoint* z) { return z->h; }
