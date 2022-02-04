/**
 * @file lqr_data.h
 * @author Brian Jackson (bjack205@gmail.com)
 * @brief LQRData type
 * @version 0.1
 * @date 2022-01-31
 *
 * @copyright Copyright (c) 2022
 *
 * @addtogroup probdef
 * @{
 */
#pragma once

#include "lqr_data.h"
#include "riccati/constants.h"
#include "slap/matrix.h"

/**
 * @brief Holds the data for a single time step of LQR
 *
 * Stores the \f$ Q, R, q, r, c \f$ values for the cost function:
 * \f[
 * \frac{1}{2} x^T Q x + q^T x + \frac{1}{2} u^T R u + r^T r + c
 * \f]
 *
 * and the \f$ A, B, d \f$ values for the dynamics:
 * \f[
 * x_{k+1} = A x_k + B u_k + d
 * \f]
 *
 * ## Construction and destruction
 * A new LQRData object is constructed using  ulqr_NewLQRData(), which must be
 * freed with a call to  ulqr_FreeLQRData().
 *
 * ## Methods
 * -  ulqr_NewLQRData()
 * -  ulqr_FreeLQRData()
 * -  ulqr_InitializeLQRData()
 * -  ulqr_CopyLQRData()
 * -  ulqr_PrintLQRData()
 *
 * ## Getters
 * The follow methods return a Matrix object wrapping the data from an LQRData object.
 * The user should NOT call FreeMatrix() on this data since it is owned by the LQRData
 * object.
 * -  ulqr_GetA()
 * -  ulqr_GetB()
 * -  ulqr_Getd()
 * -  ulqr_GetQ()
 * -  ulqr_GetR()
 * -  ulqr_Getq()
 * -  ulqr_Getr()
 *
 */
typedef struct {
  int nstates;
  int ninputs;
  Matrix Q;
  Matrix R;
  Matrix H;
  Matrix q;
  Matrix r;
  double* c;
  Matrix A;
  Matrix B;
  Matrix f;
  Matrix K;    ///< Feedback gain
  Matrix d;    ///< Feedforward gain
  Matrix P;    ///< Hessian of the cost-to-go
  Matrix p;    ///< gradient fo the cost-to-go
  Matrix Qxx;  ///< Action-value state Hessian
  Matrix Quu;  ///< Action-value control Hessian
  Matrix Qux;  ///< Action-value Hessian cross-term
  Matrix Qx;   ///< Action-value state gradient
  Matrix Qu;   ///< Action-value control gradient
  Matrix y;    ///< dual variable

  int datasize;  ///< number of doubles needed to store the data
} LQRData;

/**
 * @brief Initialize an LQRData object
 *
 * Does not allocate any new memory.
 *
 * @param lqrdata Allocated LQRData struct. Cannot be NULL.
 * @return 0 if successful
 */
enum ulqr_ReturnCode ulqr_InitializeLQRData(LQRData* lqrdata, int nstates, int ninputs,
                                            double* data);

/**
 * @brief Copies one LQRData object to another
 *
 * The two object must have equivalent dimensionality.
 *
 * @param dest Copy destination
 * @param src  Source data
 * @return 0 if successful
 */
int ulqr_CopyLQRData(LQRData* dest, LQRData* src);


int LQRDataSize(int nstates, int ninputs);

/**@} */
