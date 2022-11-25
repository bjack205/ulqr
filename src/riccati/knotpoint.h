#pragma once

#include "constants.h"
#include "riccati/constants.h"
#include "slap/matrix.h"

/**
 * @brief Stores the state, control, and time information at a single timestep / knot point
 *
 */
typedef struct {
  Matrix x;  ///< state vector
  Matrix u;  ///< control input vector
  double t;  ///< time
  double h;  ///< time step
} KnotPoint;

/**
 * @brief Initialize a KnotPoint with all of it's data
 *
 * Performs checks on the input and sets the matrices to the right size.
 * Places the state and control vectors adjacent in memory.
 *
 * @param z       Allocated knot point. Cannot be NULL.
 * @param nstates Length of state vector
 * @param ninputs Length of input vector
 * @param data    Memory to use for state and control vectors.
 *                Must have length of at least @p nstates + @p ninputs
 * @param t       Time, cannot be negative.
 * @param h       Time step, cannot be negative.
 * @return
 */
enum ulqr_ReturnCode ulqr_InitializeKnotPoint(KnotPoint* z, int nstates, int ninputs, double* data,
                                              double t, double h);

Matrix* ulqr_GetKnotpointState(KnotPoint* z);
Matrix* ulqr_GetKnotpointInput(KnotPoint* z);
double ulqr_GetTime(KnotPoint* z);
double ulqr_GetTimestep(KnotPoint* z);
