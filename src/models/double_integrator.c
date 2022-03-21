#include "double_integrator.h"
#include "riccati/lqr_data.h"
#include "riccati/constants.h"

#include <stdlib.h>
#include <stdio.h>

// Double Integrator Model
double A[4] = {0, 1, 0, 0};  // NOLINT
double B[2] = {0, 1};        // NOLINT
double Q[4] = {1, 0, 0, 1};  // NOLINT
double R[1] = {1.0};         // NOLINT

double q[2] = {0, 0};        // NOLINT
double r[1] = {0};           // NOLINT
double c = 0;                // NOLINT

void PopulateDoubleIntegratorModel() {
    const int nstates = 2;
    const int ninputs = 1;
    LQRData* lqrdata = (LQRData*)malloc(2 * sizeof(LQRData));

    int datasize = LQRDataSize(nstates, ninputs);
    double* data = (double*)malloc(2 * datasize * sizeof(double));
    ulqr_InitializeLQRData(lqrdata + 0, nstates, ninputs, data);

    LQRData* data0 = lqrdata + 0;
    SetLQRData(data0);

    printf("Test nstates: %i \n", data0->nstates);
}

void SetLQRData(LQRData* lqrdata) {
  slap_MatrixCopyFromArray(&lqrdata->Q, Q);
  slap_MatrixCopyFromArray(&lqrdata->R, R);
  slap_MatrixCopyFromArray(&lqrdata->q, q);
  slap_MatrixCopyFromArray(&lqrdata->r, r);
  slap_MatrixCopyFromArray(&lqrdata->A, A);
  slap_MatrixCopyFromArray(&lqrdata->B, B);
  *lqrdata->c = c;
}

int test_model() {
    return 0;
}
