/*
  An implementation of SVD from Numerical Recipes in C and Mike Erhdmann's lectures
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "svdcmp.h"

#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : - fabs(a))

static double maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1 = (a),maxarg2 = (b),(maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))

static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1 = (a),iminarg2 = (b),(iminarg1 < (iminarg2) ? (iminarg1) : iminarg2))

static double sqrarg;
#define SQR(a) ((sqrarg = (a)) == 0.0 ? 0.0 : sqrarg * sqrarg)

// calculates sqrt( a^2 + b^2 ) with decent precision

double pythag(double a, double b) {
  double absa,absb;

  absa = fabs(a);
  absb = fabs(b);

  if(absa > absb)
    return(absa * sqrt(1.0 + SQR(absb/absa)));
  else
    return(absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

/*
  Modified from Numerical Recipes in C
  Given a matrix a[nRows][nCols], svdcmp() computes its singular value 
  decomposition, A = U * W * Vt.  A is replaced by U when svdcmp 
  returns.  The diagonal matrix W is output as a vector w[nCols].
  V (not V transpose) is output as the matrix V[nCols][nCols].
*/

int svdcmp(double *a, int nRows, int nCols, double *w, double *v) {
  int flag,i,its,j,jj,k,l=0,nm=0;
  double anorm,c,f,g,h,s,scale,x,y,z,*rv1;

  rv1 = (double*)malloc(sizeof(double)*nCols);
  if(rv1 == NULL) {
  	printf("svdcmp(): Unable to allocate vector\n");
  	return(-1);
  }

  g = scale = anorm = 0.0;
  for(i=0;i<nCols;i++) {
    l = i+1;
    rv1[i] = scale*g;
    g = s = scale = 0.0;
    if(i < nRows) {
      for(k=i;k<nRows;k++)
		  scale += fabs(a[k*nCols+i]);
      if(scale) {
	for(k=i;k<nRows;k++) {
	  a[k*nCols+i] /= scale;
	  s += a[k*nCols+i] * a[k*nCols+i];
	}
	f = a[i*nCols+i];
	g = -SIGN(sqrt(s),f);
	h = f * g - s;
	a[i*nCols+i] = f - g;
	for(j=l;j<nCols;j++) {
	  for(s=0.0,k=i;k<nRows;k++)
		  s += a[k*nCols+i] * a[k*nCols+j];
	  f = s / h;
	  for(k=i;k<nRows;k++)
		  a[k*nCols+j] += f * a[k*nCols+i];
	}
	for(k=i;k<nRows;k++) 
		a[k*nCols+i] *= scale;
      }
    }
    w[i] = scale * g;
    g = s = scale = 0.0;
    if(i < nRows && i != nCols-1) {
      for(k=l;k<nCols;k++) scale += fabs(a[i*nCols+k]);
      if(scale)  {
	for(k=l;k<nCols;k++) {
	  a[i*nCols+k] /= scale;
	  s += a[i*nCols+k] * a[i*nCols+k];
	}
	f = a[i*nCols+l];
	g = - SIGN(sqrt(s),f);
	h = f * g - s;
	a[i*nCols+l] = f - g;
	for(k=l;k<nCols;k++) rv1[k] = a[i*nCols+k] / h;
	for(j=l;j<nRows;j++) {
	  for(s=0.0,k=l;k<nCols;k++) 
		  s += a[j*nCols+k] * a[i*nCols+k];
	  for(k=l;k<nCols;k++) 
		  a[j*nCols+k] += s * rv1[k];
	}
	for(k=l;k<nCols;k++) 
		a[i*nCols+k] *= scale;
      }
    }
    anorm = FMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));

    fflush(stdout);
  }

  for(i=nCols-1;i>=0;i--) {
    if(i < nCols-1) {
      if(g) {
	for(j=l;j<nCols;j++)
	  v[j*nCols+i] = (a[i*nCols+j] / a[i*nCols+l]) / g;
	for(j=l;j<nCols;j++) {
	  for(s=0.0,k=l;k<nCols;k++) s += a[i*nCols+k] * v[k*nCols+j];
	  for(k=l;k<nCols;k++) v[k*nCols+j] += s * v[k*nCols+i];
	}
      }
      for(j=l;j<nCols;j++) v[i*nCols+j] = v[j*nCols+i] = 0.0;
    }
    v[i*nCols+i] = 1.0;
    g = rv1[i];
    l = i;
    fflush(stdout);
  }

  for(i=IMIN(nRows,nCols) - 1;i >= 0;i--) {
    l = i + 1;
    g = w[i];
    for(j=l;j<nCols;j++) a[i*nCols+j] = 0.0;
    if(g) {
      g = 1.0 / g;
      for(j=l;j<nCols;j++) {
	for(s=0.0,k=l;k<nRows;k++) s += a[k*nCols+i] * a[k*nCols+j];
	f = (s / a[i*nCols+i]) * g;
	for(k=i;k<nRows;k++) a[k*nCols+j] += f * a[k*nCols+i];
      }
      for(j=i;j<nRows;j++) 
		  a[j*nCols+i] *= g;
    }
    else
      for(j=i;j<nRows;j++)
		  a[j*nCols+i] = 0.0;
    ++a[i*nCols+i];
//    printf(".");
//    fflush(stdout);
  }

  for(k=nCols-1;k>=0;k--) {
    for(its=0;its<30;its++) {
      flag = 1;
      for(l=k;l>=0;l--) {
	nm = l-1;
	if((fabs(rv1[l]) + anorm) == anorm) {
	  flag =  0;
	  break;
	}
	if((fabs(w[nm]) + anorm) == anorm) break;
      }
      if(flag) {
	c = 0.0;
	s = 1.0;
	for(i=l;i<=k;i++) {
	  f = s * rv1[i];
	  rv1[i] = c * rv1[i];
	  if((fabs(f) + anorm) == anorm) break;
	  g = w[i];
	  h = pythag(f,g);
	  w[i] = h;
	  h = 1.0 / h;
	  c = g * h;
	  s = -f * h;
	  for(j=0;j<nRows;j++) {
	    y = a[j*nCols+nm];
	    z = a[j*nCols+i];
	    a[j*nCols+nm] = y * c + z * s;
	    a[j*nCols+i] = z * c - y * s;
	  }
	}
      }
      z = w[k];
      if(l == k) {
	if(z < 0.0) {
	  w[k] = -z;
	  for(j=0;j<nCols;j++) v[j*nCols+k] = -v[j*nCols+k];
	}
	break;
      }
      if(its == 29) printf("no convergence in 30 svdcmp iterations\n");
      x = w[l];
      nm = k-1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = pythag(f,1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g,f))) - h)) / x;
      c = s = 1.0;
      for(j=l;j<=nm;j++) {
	i = j+1;
	g = rv1[i];
	y = w[i];
	h = s * g;
	g = c * g;
	z = pythag(f,h);
	rv1[j] = z;
	c = f/z;
	s = h/z;
	f = x * c + g * s;
	g = g * c - x * s;
	h = y * s;
	y *= c;
	for(jj=0;jj<nCols;jj++) {
	  x = v[jj*nCols+j];
	  z = v[jj*nCols+i];
	  v[jj*nCols+j] = x * c + z * s;
	  v[jj*nCols+i] = z * c - x * s;
	}
	z = pythag(f,h);
	w[j] = z;
	if(z) {
	  z = 1.0 / z;
	  c = f * z;
	  s = h * z;
	}
	f = c * g + s * y;
	x = c * y - s * g;
	for(jj=0;jj < nRows;jj++) {
	  y = a[jj*nCols+j];
	  z = a[jj*nCols+i];
	  a[jj*nCols+j] = y * c + z * s;
	  a[jj*nCols+i] = z * c - y * s;
	}
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
//    printf(".");
//   fflush(stdout);
  }
  
  free(rv1);
  
  return(0);
}


void svdbksb(double **u, double *w, double **v, int nRows, int nCols, double *b, double *x);
void svdbksb(double **u, double *w, double **v, int nRows, int nCols, double *b, double *x)
{
  int jj,j,i;
  double s;
  double *tmp;

  tmp = (double*)malloc(sizeof(double) * nCols);

  for(j=0;j<nCols;j++) { /* multiply b by U transpose */
    s = 0.0;
    if(w[j]) {
      for(i=0;i<nRows;i++)
	s += u[i][j] * b[i];
      s /= w[j];
    }
    tmp[j] = s;
  }

  for(j=0;j<nCols;j++) {
    s = 0.0;
    for(jj=0;jj<nCols;jj++)
      s += v[j][jj] * tmp[jj];
    x[j] = s;
  }

  free(tmp);
}