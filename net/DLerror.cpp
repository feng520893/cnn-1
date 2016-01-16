#include"DLerror.h"
#include<cstdio>
#include<cstdlib>

#ifdef _MSC_VER
#include <windows.h>
#endif

#define PK_ABORT 3

void HandleError(int errCode,const char* errorStr,const char *file,int line) 
{
	printf( "%s in %s at line %d\n", errorStr, file, line );

	size_t msg_len = strlen(errorStr ? errorStr : "") + 1024;
	char* message = (char*)malloc(msg_len);
	sprintf(message, "%s in %s at line %d\n"
		"Press \"Abort\" to terminate application.\n"
		"Press \"Retry\" to debug (if the app is running under debugger).\n"
		"Press \"Ignore\" to continue (this is not safe).\n",
		errorStr,file, line);
	int answer=0;
#ifdef _MSC_VER
	answer = MessageBox(NULL, message,"Cuda Error Handler", MB_ICONERROR|MB_ABORTRETRYIGNORE|MB_SYSTEMMODAL);
	if(answer==IDABORT)
		answer=PK_ABORT;
#endif
	if(message)
		free(message);
	if (answer == PK_ABORT)
		exit(errCode);
	throw errCode;
};

#ifndef CPU_ONLY
const char* cublasGetErrorString(cublasStatus_t status)
{
	switch(status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
	}
	return "unknown error";
}

const char* curandGetErrorString(curandStatus_t status)
{
	switch(status)
	{
	case CURAND_STATUS_SUCCESS :return "No errors";
    case CURAND_STATUS_VERSION_MISMATCH :return "Header file and linked library version do not match";
    case CURAND_STATUS_NOT_INITIALIZED :return "Generator not initialized";
    case CURAND_STATUS_ALLOCATION_FAILED :return "Memory allocation failed";
    case CURAND_STATUS_TYPE_ERROR :return "Generator is wrong type";
    case CURAND_STATUS_OUT_OF_RANGE :return "Argument out of range";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE :return "Length requested is not a multple of dimension";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED :return "GPU does not have double precision required by MRG32k3a";
    case CURAND_STATUS_LAUNCH_FAILURE :return "Kernel launch failure";
    case CURAND_STATUS_PREEXISTING_FAILURE :return "Preexisting failure on library entry";
    case CURAND_STATUS_INITIALIZATION_FAILED :return "Initialization of CUDA failed";
    case CURAND_STATUS_ARCH_MISMATCH :return "Architecture mismatch, GPU does not support requested feature";
	case CURAND_STATUS_INTERNAL_ERROR :return "Internal library error";
	}
	return "unknown error";
}

#endif