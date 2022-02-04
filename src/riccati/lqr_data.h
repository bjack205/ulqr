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
  Matrix x;    ///< state vector
  Matrix u;    ///< control vector
  Matrix y;    ///< dual variable

  int datasize;  ///< number of doubles needed to store the data
  bool _isowner;
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

Matrix* ulqr_GetA(LQRData* lqrdata);  ///< @brief Get (n,n) state transition matrix
Matrix* ulqr_GetB(LQRData* lqrdata);  ///< @brief Get (n,m) control input matrix
Matrix* ulqr_Getd(LQRData* lqrdata);  ///< @brief Get (n,) affine dynamice term
Matrix* ulqr_GetQ(LQRData* lqrdata);  ///< @brief Get state cost Hessian
Matrix* ulqr_GetR(LQRData* lqrdata);  ///< @brief Get control cost Hessian
Matrix* ulqr_GetH(LQRData* lqrdata);  ///< @brief Get cost Hessian cross-term (m,n)
Matrix* ulqr_Getq(LQRData* lqrdata);  ///< @brief Get affine state cost
Matrix* ulqr_Getr(LQRData* lqrdata);  ///< @brief Get affine control cost
double ulqr_Getc(LQRData* lqrdata);  ///< @brief Get cost constant 

Matrix* ulqr_GetFeedbackGain(LQRData* lqrdata);  ///< @brief Get (m,n) feedback gain 
Matrix* ulqr_GetFeedforwardGain(LQRData* lqrdata);  ///< @brief Get (m,) feedforward gain 
Matrix* ulqr_GetCostToGoHessian(LQRData* lqrdata);  ///< @brief Get (n,n) Hessian of the cost-to-go
Matrix* ulqr_GetCostToGoGradient(LQRData* lqrdata);  ///< @brief Get (n,) Gradient of the cost-to-go 
Matrix* ulqr_GetQxx(LQRData* lqrdata);  ///< @brief Get (n,n) Action-value state Hessian 
Matrix* ulqr_GetQuu(LQRData* lqrdata);  ///< @brief Get (m,m) Action-value control Hessian  
Matrix* ulqr_GetQux(LQRData* lqrdata);  ///< @brief Get (m,n) Action-value Hessian cross-term
Matrix* ulqr_GetQx(LQRData* lqrdata);  ///< @brief Get (n,) Action-value state gradient
Matrix* ulqr_GetQu(LQRData* lqrdata);  ///< @brief Get (m,) Action-value conrol gradient 
Matrix* ulqr_GetState(LQRData* lqrdata);  ///< @brief Get (n,) state vector 
Matrix* ulqr_GetControl(LQRData* lqrdata);  ///< @brief Get (m,) control vector 
Matrix* ulqr_GetDual(LQRData* lqrdata);  ///< @brief Get (n,n) dual vector 

int LQRDataSize(int nstates, int ninputs);

/**@} */
