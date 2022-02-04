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
  double* d = B + nstates * ninputs;

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
  lqrdata->d.data = d;

  lqrdata->Q.rows = nstates;
  lqrdata->Q.cols = nstates;
  lqrdata->H.rows = ninputs;
  lqrdata->H.cols = nstates;
  lqrdata->R.rows = ninputs;
  lqrdata->R.cols = ninputs;
  lqrdata->q.rows = nstates;
  lqrdata->q.cols = 1;
  lqrdata->r.rows = ninputs;
  lqrdata->r.cols = 1;

  lqrdata->A.rows = nstates;
  lqrdata->A.cols = nstates;
  lqrdata->B.rows = nstates;
  lqrdata->B.cols = ninputs;
  lqrdata->d.rows = nstates;
  lqrdata->d.cols = 1;

  lqrdata->datasize = total_size;
  lqrdata->_isowner = isowner;
  return lqrdata;
}

int ulqr_FreeLQRData(LQRData** lqrdata_ptr) {
  LQRData* lqrdata = *lqrdata_ptr; 
  if (!lqrdata) { return -1; }
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

Matrix ulqr_GetA(LQRData* lqrdata) { return lqrdata->A; }
Matrix ulqr_GetB(LQRData* lqrdata) { return lqrdata->B; }
Matrix ulqr_Getd(LQRData* lqrdata) { return lqrdata->d; }
Matrix ulqr_GetQ(LQRData* lqrdata) { return lqrdata->Q; }
Matrix ulqr_GetR(LQRData* lqrdata) { return lqrdata->R; }
Matrix ulqr_GetH(LQRData* lqrdata) { return lqrdata->H; }
Matrix ulqr_Getq(LQRData* lqrdata) { return lqrdata->q; }
Matrix ulqr_Getr(LQRData* lqrdata) { return lqrdata->r; }
double ulqr_Getc(LQRData* lqrdata) { return *lqrdata->c; }

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
  Matrix mat = ulqr_GetQ(lqrdata);
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

int LQRDataSize(int nstates, int ninputs) {
  int cost_size = (nstates + 1) * nstates + (ninputs + 1) * ninputs + nstates * ninputs +
                  1;                                                    // Q,R,q,r,c
  int dynamics_size = nstates * nstates + nstates * ninputs + nstates;  // A,B,d
  int total_size = cost_size + dynamics_size;
  return total_size;
}
