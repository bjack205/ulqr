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
  double* Q;
  double* R;
  double* H;
  double* q;
  double* r;
  double* c;
  double* A;
  double* B;
  double* d;
  int datasize;  ///< number of doubles needed to store the data
  bool isowner;
} LQRData;

/**
 * @brief Copy data into an initialized LQRData structure
 *
 * Does not allocate any new memory.
 *
 * @param lqrdata Initialized LQRData struct
 * @param Q       Diagonal of state cost Hessian
 * @param R       Diagonal of control cost Hessian
 * @param H       Cost Hessian cross-term
 * @param q       State cost affine term
 * @param r       Control cost affine term
 * @param c       Constant cost term
 * @param A       Dynamics state matrix
 * @param B       Dynamics control matrix
 * @param d       Dynamics affine term
 * @return 0 if successful
 */
int ulqr_InitializeLQRData(LQRData* lqrdata, double* Q, double* R, double* H, double* q,
                           double* r, double c, double* A, double* B, double* d);

/**
 * @brief Allocate memory for a new LQRData structure
 *
 * Must be paired with a single call to ulqr_FreeLQRData().
 *
 * @param nstates Length of the state vector
 * @param ninputs Number of control inputs
 * @param data    Pointer to memory where the data should be stored. 
 *                If NULL, the data will be allocated by this function,
 *                otherwise the data is assumed to be owned by the caller.
 * @return 0 if successful
 */
LQRData* ulqr_NewLQRData(int nstates, int ninputs, double* data);

/**
 * @brief Free the memory for and LQRData object
 *
 * @param lqrdata Address of pointer of initialized LQRData object
 * @post lqrdata = NULL
 * @return 0 if successful
 */
int ulqr_FreeLQRData(LQRData** lqrdata);


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

Matrix ulqr_GetA(LQRData* lqrdata);  ///< @brief Get (n,n) state transition matrix
Matrix ulqr_GetB(LQRData* lqrdata);  ///< @brief Get (n,m) control input matrix
Matrix ulqr_Getd(LQRData* lqrdata);  ///< @brief Get (n,) affine dynamice term
Matrix ulqr_GetQ(LQRData* lqrdata);  ///< @brief Get state cost Hessian
Matrix ulqr_GetR(LQRData* lqrdata);  ///< @brief Get control cost Hessian
Matrix ulqr_GetH(LQRData* lqrdata);  ///< @brief Get cost Hessian cross-term (m,n)
Matrix ulqr_Getq(LQRData* lqrdata);  ///< @brief Get affine state cost
Matrix ulqr_Getr(LQRData* lqrdata);  ///< @brief Get affine control cost

/**
 * @brief Prints the data contained in LQRData
 *
 * Cost data is printed in rows and dynamics data is printed as normal matrices.
 *
 * @param lqrdata
 */
void ulqr_PrintLQRData(LQRData* lqrdata);

/**@} */
