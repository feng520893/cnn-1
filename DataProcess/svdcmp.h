
// calculates sqrt( a^2 + b^2 ) with decent precision
double pythag(double a, double b);


/*
  Modified from Numerical Recipes in C
  Given a matrix a[nRows][nCols], svdcmp() computes its singular value 
  decomposition, A = U * W * Vt.  A is replaced by U when svdcmp 
  returns.  The diagonal matrix W is output as a vector w[nCols].
  V (not V transpose) is output as the matrix V[nCols][nCols].
*/
int svdcmp(double *a, int nRows, int nCols, double *w, double *v);
