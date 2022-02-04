#include "lqr_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "riccati/constants.h"
#include "slap/matrix.h"

enum ulqr_ReturnCode ulqr_InitializeLQRData(LQRData* lqrdata, int nstates, int ninputs,
                                            double* data) {
  if (nstates < 1 || ninputs < 1) {
    printf("ERROR: nstates and ninputs must be positive integers.\n");
    return kBadInput;
  }
  if (!lqrdata) {
    printf("ERROR: Pointer to lqrdata cannot be NULL.\n");
    return kBadInput;
  }
  if (!data) {
    printf("ERROR: Pointer to data cannot be NULL when initializing LQRData.\n");
    return kBadInput;
  }

  // Assign the memory into chunks
  double* Q = data;
  double* R = Q + nstates * nstates;
  double* H = R + ninputs * ninputs;
  double* q = H + nstates * ninputs;
  double* r = q + nstates;
  double* c = r + ninputs;
  double* A = c + 1;
  double* B = A + nstates * nstates;
  double* f = B + nstates * ninputs;
  double* K = f + nstates;
  double* d = K + nstates * ninputs;
  double* P = d + ninputs;
  double* p = P + nstates * nstates;
  double* Qxx = p + nstates;
  double* Quu = Qxx + nstates * nstates;
  double* Qux = Quu + ninputs * ninputs;
  double* Qx = Qux + ninputs * nstates;
  double* Qu = Qx + nstates;
  double* y = Qu + ninputs;

  // Initialize the struct
  lqrdata->nstates = nstates;
  lqrdata->ninputs = ninputs;
  lqrdata->Q.data = Q;
  lqrdata->R.data = R;
  lqrdata->H.data = H;
  lqrdata->q.data = q;
  lqrdata->r.data = r;
  lqrdata->c = c;
  lqrdata->A.data = A;
  lqrdata->B.data = B;
  lqrdata->f.data = f;
  lqrdata->K.data = K;
  lqrdata->d.data = d;
  lqrdata->P.data = P;
  lqrdata->p.data = p;
  lqrdata->Qxx.data = Qxx;
  lqrdata->Quu.data = Quu;
  lqrdata->Qux.data = Qux;
  lqrdata->q.data = q;
  lqrdata->r.data = r;
  lqrdata->y.data = y;
  lqrdata->datasize = LQRDataSize(nstates, ninputs);

  // Set matrix sizes
  slap_SetMatrixSize(&lqrdata->Q, nstates, nstates);
  slap_SetMatrixSize(&lqrdata->R, ninputs, ninputs);
  slap_SetMatrixSize(&lqrdata->H, ninputs, nstates);
  slap_SetMatrixSize(&lqrdata->q, nstates, 1);
  slap_SetMatrixSize(&lqrdata->r, ninputs, 1);
  slap_SetMatrixSize(&lqrdata->A, nstates, nstates);
  slap_SetMatrixSize(&lqrdata->B, nstates, ninputs);
  slap_SetMatrixSize(&lqrdata->f, nstates, 1);
  slap_SetMatrixSize(&lqrdata->K, ninputs, nstates);
  slap_SetMatrixSize(&lqrdata->d, ninputs, 1);
  slap_SetMatrixSize(&lqrdata->P, nstates, nstates);
  slap_SetMatrixSize(&lqrdata->p, nstates, 1);
  slap_SetMatrixSize(&lqrdata->Qxx, nstates, nstates);
  slap_SetMatrixSize(&lqrdata->Quu, ninputs, ninputs);
  slap_SetMatrixSize(&lqrdata->Qux, ninputs, nstates);
  slap_SetMatrixSize(&lqrdata->Qx, nstates, 1);
  slap_SetMatrixSize(&lqrdata->Qu, ninputs, 1);
  slap_SetMatrixSize(&lqrdata->y, nstates, 1);

  return 0;
}

int ulqr_CopyLQRData(LQRData* dest, LQRData* src) {
  if (dest->nstates != src->nstates || dest->ninputs != src->ninputs) {
    fprintf(stderr, "Can't copy LQRData of different sizes: (%d,%d) and (%d,%d).\n",
            dest->nstates, dest->ninputs, src->nstates, src->ninputs);
    return -1;
  }
  int total_size = src->datasize;
  memcpy(dest->Q.data, src->Q.data, total_size * sizeof(double));
  return 0;
}

Matrix* ulqr_GetA(LQRData* lqrdata) { return &lqrdata->A; }
Matrix* ulqr_GetB(LQRData* lqrdata) { return &lqrdata->B; }
Matrix* ulqr_Getf(LQRData* lqrdata) { return &lqrdata->f; }
Matrix* ulqr_GetQ(LQRData* lqrdata) { return &lqrdata->Q; }
Matrix* ulqr_GetR(LQRData* lqrdata) { return &lqrdata->R; }
Matrix* ulqr_GetH(LQRData* lqrdata) { return &lqrdata->H; }
Matrix* ulqr_Getq(LQRData* lqrdata) { return &lqrdata->q; }
Matrix* ulqr_Getr(LQRData* lqrdata) { return &lqrdata->r; }
double ulqr_Getc(LQRData* lqrdata) { return *lqrdata->c; }

Matrix* ulqr_GetFeedbackGain(LQRData* lqrdata) { return &lqrdata->K; }
Matrix* ulqr_GetFeedforwardGain(LQRData* lqrdata) { return &lqrdata->d; }
Matrix* ulqr_GetCostToGoHessian(LQRData* lqrdata) { return &lqrdata->P; }
Matrix* ulqr_GetCostToGoGradient(LQRData* lqrdata) { return &lqrdata->p; }
Matrix* ulqr_GetQxx(LQRData* lqrdata) { return &lqrdata->Qxx; }
Matrix* ulqr_GetQuu(LQRData* lqrdata) { return &lqrdata->Quu; }
Matrix* ulqr_GetQux(LQRData* lqrdata) { return &lqrdata->Qux; }
Matrix* ulqr_GetQx(LQRData* lqrdata) { return &lqrdata->Qx; }
Matrix* ulqr_GetQu(LQRData* lqrdata) { return &lqrdata->Qu; }
Matrix* ulqr_GetDual(LQRData* lqrdata) { return &lqrdata->y; }

int LQRDataSize(int nstates, int ninputs) {
  int cost_size = (nstates + 1) * nstates + (ninputs + 1) * ninputs + nstates * ninputs +
                  1;                                                    // Q,R,q,r,c
  int dynamics_size = nstates * nstates + nstates * ninputs + nstates;  // A,B,d
  int gains_size = ninputs * (nstates + 1);
  int ctg_size = nstates * (nstates + 1);
  int action_value_size = cost_size - 1;
  int vec_size = nstates;
  int total_size =
      cost_size + dynamics_size + gains_size + ctg_size + action_value_size + vec_size;
  return total_size;
}
