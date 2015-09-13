#ifndef DL_ERROR_H
#define DL_ERROR_H
#include<cuda_runtime_api.h>
#include<cublas_v2.h>
#include <curand.h>

#define DL_StsInternal     -1  /* internal error (bad state)      */

void HandleError(int errCode,const char* errorStr,const char *file,int line);
const char* cublasGetErrorString(cublasStatus_t status);
const char* curandGetErrorString(curandStatus_t status);

//该判定Debug和Release都有效
#define CUDA_ERROR(err){\
	if(err!=cudaSuccess)\
	HandleError(err,cudaGetErrorString( err ), __FILE__, __LINE__ );}

//该判定Debug和Release都有效
#define CUBLAS_ERROR(err ){\
	if(err!=CUBLAS_STATUS_SUCCESS)\
	HandleError(err,cublasGetErrorString( err ), __FILE__, __LINE__ );}

#define CURAND_ERROR(err ){\
	if(err!=CURAND_STATUS_SUCCESS)\
	HandleError(err,curandGetErrorString( err ), __FILE__, __LINE__ );}

#define DL_ASSER(Condition)											\
{                                                                       \
    if (!(Condition))													\
        HandleError(DL_StsInternal, "Assertion: " #Condition " failed",__FILE__, __LINE__);	\
}

#endif