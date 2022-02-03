#include "lqr_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ulqr_InitializeLQRData(LQRData* lqrdata, double* Q, double* R, double* H, double* q, double* r,
                           double c, double* A, double* B, double* d) {
  if (!lqrdata || !Q || !R || !H || !q || !r || !A || !B || !d) { 
    return -1; 
  }
  int nstates = lqrdata->nstates;
  int ninputs = lqrdata->ninputs;
  memcpy(lqrdata->Q, Q, nstates * nstates * sizeof(double));
  memcpy(lqrdata->R, R, ninputs * ninputs * sizeof(double));
  memcpy(lqrdata->H, H, ninputs * nstates * sizeof(double));
  memcpy(lqrdata->q, q, nstates * sizeof(double));
  memcpy(lqrdata->r, r, ninputs * sizeof(double));
  *lqrdata->c = c;
  memcpy(lqrdata->A, A, nstates * nstates * sizeof(double));
  memcpy(lqrdata->B, B, nstates * ninputs * sizeof(double));
  memcpy(lqrdata->d, d, nstates * sizeof(double));
  return 0;
}

LQRData* ulqr_NewLQRData(int nstates, int ninputs) {
  int cost_size = (nstates + 1) * nstates + (ninputs + 1) * ninputs + nstates * ninputs +
                  1;                                                    // Q,R,q,r,c
  int dynamics_size = nstates * nstates + nstates * ninputs + nstates;  // A,B,d
  int total_size = cost_size + dynamics_size;
  double* data = (double*)malloc(total_size * sizeof(double));
  double* Q = data;
  double* R = Q + nstates * nstates;
  double* H = R + ninputs * ninputs;
  double* q = H + nstates * ninputs;
  double* r = q + nstates;
  double* c = r + ninputs;
  double* A = c + 1;
  double* B = A + nstates * nstates;
  double* d = B + nstates * ninputs;
  LQRData* lqrdata = (LQRData*)malloc(sizeof(LQRData));
  lqrdata->nstates = nstates;
  lqrdata->ninputs = ninputs;
  lqrdata->Q = Q;
  lqrdata->R = R;
  lqrdata->H = H;
  lqrdata->q = q;
  lqrdata->r = r;
  lqrdata->c = c;
  lqrdata->A = A;
  lqrdata->B = B;
  lqrdata->d = d;
  return lqrdata;
}

int ulqr_FreeLQRData(LQRData** lqrdata_ptr) {
  LQRData* lqrdata = *lqrdata_ptr; 
  if (!lqrdata) { return -1; }
  if (lqrdata->Q) {
    free(lqrdata->Q);  // This points to the beginning of the allocated memory block
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
  int nstates = dest->nstates;
  int ninputs = dest->ninputs;
  int cost_size = (nstates + 1) * nstates + (ninputs + 1) * ninputs + nstates * ninputs +
                  1;                                                    // Q,R,q,r,c
  int dynamics_size = nstates * nstates + nstates * ninputs + nstates;  // A,B,d
  int total_size = cost_size + dynamics_size;
  memcpy(dest->Q, src->Q, total_size * sizeof(double));
  return 0;
}

Matrix ulqr_GetA(LQRData* lqrdata) {
  Matrix mat = {lqrdata->nstates, lqrdata->nstates, lqrdata->A};
  return mat;
}

Matrix ulqr_GetB(LQRData* lqrdata) {
  Matrix mat = {lqrdata->nstates, lqrdata->ninputs, lqrdata->B};
  return mat;
}

Matrix ulqr_Getd(LQRData* lqrdata) {
  Matrix mat = {lqrdata->nstates, 1, lqrdata->d};
  return mat;
}

Matrix ulqr_GetQ(LQRData* lqrdata) {
  Matrix mat = {lqrdata->nstates, lqrdata->nstates, lqrdata->Q};
  return mat;
}

Matrix ulqr_Getq(LQRData* lqrdata) {
  Matrix mat = {lqrdata->nstates, 1, lqrdata->q};
  return mat;
}

Matrix ulqr_GetR(LQRData* lqrdata) {
  Matrix mat = {lqrdata->ninputs, lqrdata->ninputs, lqrdata->R};
  return mat;
}

Matrix ulqr_Getr(LQRData* lqrdata) {
  Matrix mat = {lqrdata->ninputs, 1, lqrdata->r};
  return mat;
}

Matrix ulqr_GetH(LQRData* lqrdata) {
  Matrix mat = {lqrdata->ninputs, lqrdata->nstates, lqrdata->H};
  return mat;
}

void PrintAsRow(Matrix* mat) {
  printf("[");
  for (int i = 0; i < slap_MatrixNumElements(mat); ++i) {
    printf("%6.2f ", mat->data[i]);
  }
  printf("]\n");
}

void ulqr_PrintLQRData(LQRData* lqrdata) {
  // clang-format off
  printf("LQR Data with n=%d, m=%d:\n", lqrdata->nstates, lqrdata->ninputs);
  Matrix mat =  ulqr_GetQ(lqrdata);
  printf("Q = "); PrintAsRow(&mat);
  mat =  ulqr_GetR(lqrdata);
  printf("R = "); PrintAsRow(&mat);
  mat =  ulqr_Getq(lqrdata);
  printf("q = "); PrintAsRow(&mat);
  mat =  ulqr_Getr(lqrdata);
  printf("r = "); PrintAsRow(&mat);
  printf("c = %f\n", *lqrdata->c);
  printf("A:\n");
  mat =  ulqr_GetA(lqrdata);
  slap_PrintMatrix(&mat);
  printf("B:\n");
  mat =  ulqr_GetB(lqrdata);
  slap_PrintMatrix(&mat);
  mat =  ulqr_Getd(lqrdata);
  printf("d = "); PrintAsRow(&mat);
  // clang-format on
}
