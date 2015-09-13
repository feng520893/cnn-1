#ifndef MAT_CUDA_H
#define MAT_CUDA_H
#include<cuda_runtime.h>
#include<cublas_v2.h>

void matrixMul(double * x,int rowsX,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ);

#endif