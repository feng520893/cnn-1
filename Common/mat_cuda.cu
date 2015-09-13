
#include"mat_cuda.cuh"
#include<stdio.h>

cublasHandle_t g_handle=NULL;

void matrixMul(double * x,int rowsX,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ)
{
	cublasStatus_t ret=CUBLAS_STATUS_SUCCESS;
	if(g_handle==NULL)
		ret = cublasCreate(&g_handle);
	if(ret != CUBLAS_STATUS_SUCCESS)
		printf( "cublasSgemm returned error code");

 	cublasStatus_t stat; 
 	double alpha = 1.0; 
 	double beta = 0.0; 
 	stat = cublasDgemm( 
 		g_handle,  
 		CUBLAS_OP_N, 
 		CUBLAS_OP_N, 
 		colsY, 
		rowsX, 
 		rowsY, 
 		&alpha, 
 		y, 
 		colsY, 
 		x, 
 		colsX, 
 		&beta, 
 		z, 
 		colsZ); 
 	cudaDeviceSynchronize(); 
 	if(stat != CUBLAS_STATUS_SUCCESS) 
	{ 
 		printf("matrixMulTA cublasSgemm error\n"); 
		exit(0); 
	} 
}