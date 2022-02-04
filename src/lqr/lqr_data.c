#include "lqr_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "slap/matrix.h"

int ulqr_InitializeLQRData(LQRData* lqrdata, double* Q, double* R, double* H, double* q,
                           double* r, double c, double* A, double* B, double* d) {
  if (!lqrdata || !Q || !R || !H || !q || !r || !A || !B || !d) {
    return -1;
  }
  int nstates = lqrdata->nstates;
  int ninputs = lqrdata->ninputs;
  memcpy(lqrdata->Q.data, Q, nstates * nstates * sizeof(double));
  memcpy(lqrdata->R.data, R, ninputs * ninputs * sizeof(double));
  memcpy(lqrdata->H.data, H, ninputs * nstates * sizeof(double));
  memcpy(lqrdata->q.data, q, nstates * sizeof(double));
  memcpy(lqrdata->r.data, r, ninputs * sizeof(double));
  *lqrdata->c = c;
  memcpy(lqrdata->A.data, A, nstates * nstates * sizeof(double));
  memcpy(lqrdata->B.data, B, nstates * ninputs * sizeof(double));
  memcpy(lqrdata->d.data, d, nstates * sizeof(double));
  return 0;
}

LQRData* ulqr_NewLQRData(int nstates, int ninputs, double* data) {
  if (nstates < 1 || ninputs < 1) {
    printf("ERROR: nstates and ninputs must be positive integers.\n");
    return NULL;
  }
  int total_size = LQRDataSize(nstates, ninputs);

  bool isowner = data == NULL;
  if (isowner) {
    data = (double*)malloc(total_size * sizeof(double));
  }
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
  double* x = Qu + ninputs;
  double* u = x + nstates;
  double* y = u + ninputs;

  LQRData* lqrdata = (LQRData*)malloc(sizeof(LQRData));
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
  lqrdata->x.data = x;
  lqrdata->u.data = u;
  lqrdata->y.data = y;

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
  slap_SetMatrixSize(&lqrdata->x, nstates, 1);
  slap_SetMatrixSize(&lqrdata->u, ninputs, 1);
  slap_SetMatrixSize(&lqrdata->y, nstates, 1);

  lqrdata->datasize = total_size;
  lqrdata->_isowner = isowner;
  return lqrdata;
}

int ulqr_FreeLQRData(LQRData** lqrdata_ptr) {
  LQRData* lqrdata = *lqrdata_ptr;
  if (!lqrdata) {
    return -1;
  }
  if (lqrdata->_isowner && lqrdata->Q.data) {
    free(lqrdata->Q.data);  // This points to the beginning of the allocated memory block
  }
  free(lqrdata);
  *lqrdata_ptr = NULL;
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
Matrix* ulqr_Getd(LQRData* lqrdata) { return &lqrdata->d; }
Matrix* ulqr_GetQ(LQRData* lqrdata) { return &lqrdata->Q; }
Matrix* ulqr_GetR(LQRData* lqrdata) { return &lqrdata->R; }
Matrix* ulqr_GetH(LQRData* lqrdata) { return &lqrdata->H; }
Matrix* ulqr_Getq(LQRData* lqrdata) { return &lqrdata->q; }
Matrix* ulqr_Getr(LQRData* lqrdata) { return &lqrdata->r; }
double ulqr_Getc(LQRData* lqrdata) { return *lqrdata->c; }

int LQRDataSize(int nstates, int ninputs) {
  int cost_size = (nstates + 1) * nstates + (ninputs + 1) * ninputs + nstates * ninputs +
                  1;                                                    // Q,R,q,r,c
  int dynamics_size = nstates * nstates + nstates * ninputs + nstates;  // A,B,d
  int gains_size = ninputs * (nstates + 1);
  int ctg_size = nstates * (nstates + 1);
  int action_value_size = cost_size - 1;
  int vec_size = 2 * nstates + ninputs;
  int total_size =
      cost_size + dynamics_size + gains_size + ctg_size + action_value_size + vec_size;
  return total_size;
}
