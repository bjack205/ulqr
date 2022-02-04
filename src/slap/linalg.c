#include "linalg.h"

#include "math.h"
#include "slap/matrix.h"
#include "stdio.h"

int slap_MatrixAddition(Matrix* A, Matrix* B, double alpha) {
  for (int i = 0; i < slap_MatrixNumElements(A); ++i) {
    B->data[i] += alpha * A->data[i];
  }
  return 0;
}

int slap_MatrixScale(Matrix* A, double alpha) {
  for (int i = 0; i < slap_MatrixNumElements(A); ++i) {
    A->data[i] *= alpha;
  }
  return 0;
}

int slap_MatrixMultiply(Matrix* A, Matrix* B, Matrix* C, bool tA, bool tB, double alpha,
                        double beta) {
  int n;
  int m;
  if (tA) {
    n = A->cols;
    m = A->rows;
  } else {
    n = A->rows;
    m = A->cols;
  }
  int p = tB ? B->rows : B->cols;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      double* Cij = slap_MatrixGetElement(C, i, j);
      *Cij *= beta;
      for (int k = 0; k < m; ++k) {
        double Aik = *slap_MatrixGetElementTranspose(A, i, k, tA);
        double Bkj = *slap_MatrixGetElementTranspose(B, k, j, tB);
        *Cij += alpha * Aik * Bkj;
      }
    }
  }
  return 0;
}

int slap_SymmetricMatrixMultiply(Matrix* Asym, Matrix* B, Matrix* C, double alpha,
                                 double beta) {
  int n;
  int m;
  bool tA = false;
  bool tB = false;
  if (tA) {
    n = Asym->cols;
    m = Asym->rows;
  } else {
    n = Asym->rows;
    m = Asym->cols;
  }
  int p = tB ? B->rows : B->cols;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      double* Cij = slap_MatrixGetElement(C, i, j);
      *Cij *= beta;
      for (int k = 0; k < m; ++k) {
        int row = i;
        int col = k;
        if (i < k) {
          row = k;
          col = i;
        }
        double Aik = *slap_MatrixGetElement(Asym, row, col);
        double Bkj = *slap_MatrixGetElement(B, k, j);
        *Cij += alpha * Aik * Bkj;
      }
    }
  }
  return 0;
  return 0;
}

int slap_AddDiagonal(Matrix* A, double alpha) {
  int n = A->rows;
  for (int i = 0; i < n; ++i) {
    double* Aii = slap_MatrixGetElement(A, i, i);
    *Aii += alpha;
  }
  return 0;
}

int slap_CholeskyFactorize(Matrix* A) {
  int n = A->rows;
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < j; ++k) {
      for (int i = j; i < n; ++i) {
        double* Aij = slap_MatrixGetElement(A, i, j);
        double Aik = *slap_MatrixGetElement(A, i, k);
        double Ajk = *slap_MatrixGetElement(A, j, k);
        *Aij -= Aik * Ajk;
      }
    }
    double Ajj = *slap_MatrixGetElement(A, j, j);
    if (Ajj <= 0) {
      return slap_kCholeskyFail;
    }
    double ajj = sqrt(Ajj);

    for (int i = j; i < n; ++i) {
      double* Aij = slap_MatrixGetElement(A, i, j);
      *Aij /= ajj;
    }
  }
  return slap_kCholeskySuccess;
}

int slap_LowerTriBackSub(Matrix* L, Matrix* b, bool istransposed) {
  int n = b->rows;
  int m = b->cols;
  for (int j_ = 0; j_ < n; ++j_) {
    int j = istransposed ? n - j_ - 1 : j_;
    for (int k = 0; k < m; ++k) {
      double* xjk = slap_MatrixGetElement(b, j, k);
      double Ljj = *slap_MatrixGetElement(L, j, j);
      *xjk /= Ljj;

      for (int i_ = j_ + 1; i_ < n; ++i_) {
        int i = istransposed ? i_ - (j_ + 1) : i_;
        double* xik = slap_MatrixGetElement(b, i, k);
        double Lij = *slap_MatrixGetElementTranspose(L, i, j, istransposed);
        *xik -= Lij * (*xjk);
      }
    }
  }
  return 0;
}

int slap_CholeskySolve(Matrix* L, Matrix* b) {
  slap_LowerTriBackSub(L, b, 0);
  slap_LowerTriBackSub(L, b, 1);
  return 0;
}

double slap_TwoNorm(const Matrix* M) {
  if (!M) {
    return -1;
  }
  double norm = 0.0;
  for (int i = 0; i < slap_MatrixNumElements(M); ++i) {
    double x = M->data[i];
    norm += x * x;
  }
  return sqrt(norm);
}

double slap_OneNorm(const Matrix* M) {
  if (!M) {
    return -1;
  }
  double norm = 0.0;
  for (int i = 0; i < slap_MatrixNumElements(M); ++i) {
    double x = M->data[i];
    norm += fabs(x);
  }
  return sqrt(norm);
}
