#include "slap/linalg.h"

#include "simpletest/simpletest.h"
#include "slap/matrix.h"

#ifdef USE_EIGEN
#include "eigen_c/eigen_c.h"
#endif

int MatMul() {
  // Matrix-matrix
  Matrix A = slap_NewMatrix(3, 4);
  Matrix B = slap_NewMatrix(4, 5);
  Matrix C = slap_NewMatrix(3, 5);
  Matrix D = slap_NewMatrix(4, 3);
  Matrix Cans = slap_NewMatrix(3, 5);
  Matrix Bans = slap_NewMatrix(4, 5);
  Matrix Dans = slap_NewMatrix(4, 3);
  slap_MatrixSetConst(&A, 4);
  slap_MatrixSetConst(&B, 3);
  slap_MatrixSetConst(&C, 2);
  slap_MatrixSetConst(&Cans, 50);
  slap_MatrixSetConst(&Bans, 3);
  slap_MatrixSetConst(&Dans, 750);
  slap_MatrixMultiply(&A, &B, &C, 0, 0, 1.0, 1.0);
  TEST(slap_MatrixNormedDifference(&C, &Cans) < 1e-6);

  slap_MatrixMultiply(&A, &C, &B, 1, 0, 1.0, -199);
  TEST(slap_MatrixNormedDifference(&B, &Bans) < 1e-6);

  slap_MatrixMultiply(&B, &C, &D, 0, 1, 1.0, 0.0);
  TEST(slap_MatrixNormedDifference(&D, &Dans) < 1e-6);

  // Matrix-vector
  double xdata[4] = {1, 2, 3, 4};
  double bdata[3] = {40, 40, 40};
  Matrix x = {4, 1, xdata};
  Matrix bans = {3, 1, bdata};
  Matrix b = slap_NewMatrix(3, 1);
  slap_MatrixMultiply(&A, &x, &b, 0, 0, 1.0, 0.0);
  TEST(slap_MatrixNormedDifference(&b, &bans) < 1e-6);

  slap_FreeMatrix(&A);
  slap_FreeMatrix(&B);
  slap_FreeMatrix(&C);
  slap_FreeMatrix(&D);
  slap_FreeMatrix(&Cans);
  slap_FreeMatrix(&Bans);
  slap_FreeMatrix(&Dans);
  slap_FreeMatrix(&b);
  return 1;
}

int SymMatMulTest() {
  // clang-format off
  double Adata[9] = {1,2,3, 4,5,6, 7,8,9};
  double Bdata[6] = {2,4,6, 8,6,4};
  double Cdata[6] = {3,6,9, 12,11,10};
  double Ddata[6] = {34,72,102, 56,92,116};
  // clang-format on
  Matrix A = {3, 3, Adata};
  Matrix B = {3, 2, Bdata};
  Matrix C = {3, 2, Cdata};
  Matrix D = {3, 2, Ddata};
  slap_SymmetricMatrixMultiply(&A, &B, &C, 1.0, 2.0);
  TEST(slap_MatrixNormedDifference(&C, &D) < 1e-6);
  return 1;
}

int MatAddTest() {
  // clang-format off
  double Adata[6] = {1,2,3, 4,5,6};
  double Bdata[6] = {2,4,6, 8,6,4};
  double Cdata[6] = {3,6,9, 12,11,10};
  double Ddata[6] = {1,2,3, 4,1,-2};
  // clang-format on
  Matrix A = {2, 3, Adata};
  Matrix B = {2, 3, Bdata};
  Matrix C = {2, 3, Cdata};
  Matrix D = {2, 3, Ddata};
  slap_MatrixAddition(&A, &B, 1.0);
  TEST(slap_MatrixNormedDifference(&B, &C) < 1e-6);

  slap_MatrixAddition(&A, &C, -2);
  TEST(slap_MatrixNormedDifference(&C, &D) < 1e-6);

  return 1;
}

int MatScale() {
  // clang-format off
  double Adata[6] = {1,2,3, 4,5,6};
  double Bdata[6] = {3,6,9, 12,15,18};
  Matrix A = {2, 3, Adata};
  Matrix B = {2, 3, Bdata};
  // clang-format on
  slap_MatrixScale(&A, 3);
  TEST(slap_MatrixNormedDifference(&A, &B) < 1e-6);
  return 1;
}

int CholeskyFactorizeTest() {
  int n = 10;
  Matrix A1 = slap_NewMatrix(n, n);
  Matrix A2 = slap_NewMatrix(n, n);
  Matrix A = slap_NewMatrix(n, n);
  Matrix Achol = slap_NewMatrix(n, n);
  for (int i = 0; i < n * n; ++i) {
    A1.data[i] = (i - 4) * (i + 3) / 6.0;
    A2.data[i] = A1.data[i];
  }
  slap_MatrixMultiply(&A1, &A2, &A, 1, 0, 1.0, 0.0);
  slap_AddDiagonal(&A, 1.0);
  slap_MatrixCopy(&Achol, &A);
  int res = slap_CholeskyFactorize(&Achol);
  TEST(res == slap_kCholeskySuccess);

#ifdef USE_EIGEN
  // Check answer with Eigen
  void* fact;
  eigen_CholeskyFactorize(n, A.data, &fact);
  mu_assert(MatrixNormedDifference(&A, &Achol) < 1e-6);
#endif

  // Try to factorize an indefinite matrix
  slap_MatrixMultiply(&A1, &A2, &A, 1, 0, 1.0, 0.0);
  slap_AddDiagonal(&A, -1.0);
  slap_MatrixCopy(&Achol, &A);
  res = slap_CholeskyFactorize(&Achol);
  TEST(res == slap_kCholeskyFail);

  slap_FreeMatrix(&A1);
  slap_FreeMatrix(&A2);
  slap_FreeMatrix(&A);
  slap_FreeMatrix(&Achol);
  return 1;
}

int TriBackSubTest() {
  int n = 3;
  double Ldata[9] = {1, 2, 5, 0, 1, 6, 0, 0, 7};
  double bdata[3] = {-2, 3, 10};
  double ydata[3] = {-2.0, 7.0, -3.142857142857143};
  double xdata[3] = {-19.142857142857142, 9.693877551020408, -0.4489795918367347};

  Matrix L = {n, n, Ldata};
  Matrix b = {n, 1, bdata};
  Matrix y = {n, 1, ydata};
  Matrix x = {n, 1, xdata};
  slap_LowerTriBackSub(&L, &b, 0);
  TEST(slap_MatrixNormedDifference(&b, &y) < 1e-6);

  slap_LowerTriBackSub(&L, &y, 1);
  TEST(slap_MatrixNormedDifference(&x, &y) < 1e-6);

  return 1;
}

int CholeskySolveTest() {
  int n = 10;
  int m = 1;
  Matrix A1 = slap_NewMatrix(n, n);
  Matrix A2 = slap_NewMatrix(n, n);
  Matrix A = slap_NewMatrix(n, n);
  Matrix Achol = slap_NewMatrix(n, n);
  Matrix b = slap_NewMatrix(n, m);
  Matrix x = slap_NewMatrix(n, m);
  Matrix x_eigen = slap_NewMatrix(n, m);
  for (int i = 0; i < n * n; ++i) {
    A1.data[i] = (i - 4) * (i + 3) / 6.0;
    A2.data[i] = A1.data[i];
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      slap_MatrixSetElement(&b, i, j, -i - (n + m) / 2 + j);
    }
  }

  slap_MatrixMultiply(&A1, &A2, &A, 1, 0, 1.0, 0.0);
  slap_AddDiagonal(&A, 1.0);
  slap_MatrixCopy(&Achol, &A);
  slap_MatrixCopy(&x, &b);
  slap_MatrixCopy(&x_eigen, &b);

  slap_CholeskyFactorize(&Achol);
  slap_CholeskySolve(&Achol, &x);

#ifdef USE_EIGEN
  // Check answer with Eigen
  void* fact = NULL;
  eigen_CholeskyFactorize(n, A.data, &fact);
  eigen_CholeskySolve(n, m, fact, x_eigen.data);
  mu_assert(MatrixNormedDifference(&x, &x_eigen) < 1e-6);
#endif

  slap_FreeMatrix(&A1);
  slap_FreeMatrix(&A2);
  slap_FreeMatrix(&A);
  slap_FreeMatrix(&Achol);
  slap_FreeMatrix(&b);
  slap_FreeMatrix(&x);
  slap_FreeMatrix(&x_eigen);
#ifdef USE_EIGEN
  eigen_FreeFactorization(fact);
#endif
  return 1;
}

void AllTests() {
  MatMul();
  MatAddTest();
  MatScale();
  CholeskyFactorizeTest();
  TriBackSubTest();
  CholeskySolveTest();
  SymMatMulTest();
#ifdef USE_EIGEN
  printf("Using Eigen library for comparisons.\n");
#endif
}

int main() {
  AllTests();
  PrintTestResult();
  return TestResult();
}
