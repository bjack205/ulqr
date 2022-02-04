/**
 * @file riccati_solver.h
 * @author Brian Jackson (bjack205@gmail.com)
 * @brief Basic methods for creating and using the Riccati solver
 * @version 0.1
 * @date 2022-01-30
 *
 * @copyright Copyright (c) 2022
 *
 * @addtogroup riccati
 * @{
 */
#pragma once

#include "lqr_data.h"

/**
 * @brief Solver that uses Riccati recursion to solve an LQR problem.
 *
 * Solves the generic LQR problem with affine terms using Riccati recursion and
 * a forward simulation of the linear dynamics. Assumes problems are of the following form:
 *
 * \f{align*}{
 * \underset{x_{1:N}, u_{1:N-1}}{\text{minimize}} &&& \frac{1}{2} x_N^T Q_N + x_N + q_N^T
 * x_N + \sum_{k-1}^{N-1} \frac{1}{2} x_k^T Q_k + x_k + q_k^T x_k + u_k^T R_k + u_k + r_k^T
 * u_k \\
 * \text{subject to} &&& x_{k+1} = A_k x_k + B_k u_k + f_k \\
 * &&& x_1 = x_\text{init}
 * \f}
 *
 * All the memory required by the solver
 * is initialized upon the creation of the solver to avoid any dynamic memory allocations
 * during the solve.
 *
 * ## Construction and destruction
 * Use  ulqr_NewRiccatiSolver() to initialize a new solver, which much be paired
 * with a single call to  ulqr_FreeRiccatiSolver() to free all of solver's memory.
 *
 * ## Typical Usage
 * Standard usage will typically look like the following:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * LQRProblem* lqrprob =  ulqr_ReadTestLQRProblem();  // your data here
 * RiccatiSolver* solver =  ulqr_NewRiccatiSolver(lqrprob);
 *  ulqr_SolveRiccati(solver);
 *  ulqr_PrintRiccatiSummary(solver);
 * double* soln = (double*) malloc(solver->nvars * sizeof(double));
 *  ulqr_CopyRiccatiSolution(solver, soln);
 *  ulqr_FreeRiccatiSolver();
 *  ulqr_FreeLQRProblem();
 * free(soln);
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * ## Methods
 * -  ulqr_NewRiccatiSolver()
 * -  ulqr_FreeRiccatiSolver()
 * -  ulqr_PrintRiccatiSummary()
 * -  ulqr_GetRiccatiSolution()
 * -  ulqr_CopyRiccatiSolution()
 * -  ulqr_GetRiccatiSolveTimes()
 */
typedef struct {
  // clang-format off
  int nhorizon;  ///< length of the time horizon
  int nstates;   ///< size of state vector (n)
  int ninputs;   ///< number of control inputs (m)
  int nvars;     ///< total number of decision variables, including the dual variables
  double* data;  ///< pointer to the beginning of the single block of memory allocated by the solver
  LQRData** lqrdata;  ///< LQR Problem data
  Matrix x0;    ///< Initial state
  double t_solve_ms;          ///< Total solve time in milliseconds
  double t_backward_pass_ms;  ///< Time spent in the backward pass in milliseconds
  double t_forward_pass_ms;   ///< Time spent in the forward pass in milliseconds
  // clang-format on
} RiccatiSolver;

/**
 * @brief Initialize a new Riccati solver
 *
 * Create a new Riccati solver, provided the problem data given by lqrprob.
 *
 * @param lqrprob Contains all the data to describe the LQR problem to be solved.
 * @return An initialized Riccati solver.
 */
RiccatiSolver* ulqr_NewRiccatiSolver(int nstates, int ninputs, int nhorizon);

/**
 * @brief Free the memory for a Riccati solver
 *
 * @param solver Pointer to initialized Riccati solver.
 * @post solver will be NULL
 * @return 0 if successful
 */
int ulqr_FreeRiccatiSolver(RiccatiSolver** solver);

/**
 * @brief Prints a summary of the solve
 *
 * ## Sample output
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.txt}
 * NDLQR Riccati Solve Summary
 * Solve time:    1.24 ms
 * Backward Pass: 1.13 ms (91.1 % of total)
 * Foward Pass:   0.11 ms (8.9 % of total)
 * Final error: 5.05696e-12
 * Final error after 2nd solve: 5.05696e-12
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * @pre  ulqr_SolveRiccati() has already been called
 * @param solver An initialized solver
 * @return 0 if successful
 */
int ulqr_PrintRiccatiSummary(RiccatiSolver* solver);

int ulqr_GetNumVars(RiccatiSolver* solver);



/**@} */
