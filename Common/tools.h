#ifndef TOOLS_H
#define TOOLS_H

#ifndef CPU_ONLY
#include<cuda_runtime.h>
#include <curand.h>
#include<cublas_v2.h>
#endif

#include<string>
#include<vector>
#include<time.h>

class CTools
{
public:
	enum RAND_TYPE{NORMAL,GAUSS};
	enum MAT_MUL_TYPE{NORMAL_XY,TRANSPOSE_X,TRANSPOSE_Y};
	static CTools* initialize();
	static void destroy();

	static int findDirectsOrFiles(std::string direct,std::vector<std::string>& files,const char* extension=NULL,bool bOnlyFindDirect=false);

	template<typename T>
	static int cpuRand(T* data,int dataSize,RAND_TYPE type,float rate=0.1)
	{
		srand(time(NULL));
		if(type == NORMAL)
			return m_pThis->rand<T>(data,dataSize);
		return m_pThis->randn<T>(data,dataSize,rate);
	}

#ifndef CPU_ONLY
	//判断是否有可用的CUDA
	static bool isCUDA();
	//生成单精度浮点随机数
	static curandStatus_t cudaRandF(float* data,unsigned int dataSize,RAND_TYPE type,float mean=0.0,float stddev=0.1);
   //生成双精度浮点随机数
	static curandStatus_t cudaRandD(double* data,unsigned int dataSize,RAND_TYPE type,float mean=0.0,float stddev=0.1);
	//双精度浮点矩阵相乘
	static void cudaMatrixMulD(double* x,int rowsX,int colsX,double* y,int rowsY,int colsY,double* z,int colsZ,MAT_MUL_TYPE type);
#endif

private:
	static CTools* m_pThis;

#ifndef CPU_ONLY
	curandGenerator_t m_hGen;
	cublasHandle_t    m_hCublas;
/*
	因为传入数据以行为主，而CUDA以列为主,结果要转置
	TA==>     cpu: z= x * y' cuda:z'=x'*y;
	TB==>     cpu: z= x'* y  cuda:z'=x*y';
	NORMAL==> cpu: z= x * y  cuda:z'=y*x;
*/
    cublasStatus_t cudaMatrixDMulTA(double* x,int rowsX,int colsX,double* y,int rowsY,int colsY,double* z,int colsZ);
    cublasStatus_t cudaMatrixDMulTB(double* x,int colsX,double* y,int rowsY,int colsY,double* z,int colsZ);
    cublasStatus_t cudaMatrixDMul(double* x,int rowsX,int colsX,double* y,int rowsY,int colsY,double* z,int colsZ);
#endif

	CTools()
	{
#ifndef CPU_ONLY
		m_hGen=NULL;
		m_hCublas=NULL;
#endif
	};
	~CTools(){};

	//高斯随机分布
	template<typename T>
	int randn(T* data,int dataSize,float rate=0.1)
	{
		for(int i=0;i<dataSize;i++)
		{
			T V1, V2, S;
			int phase = 0;
			T X;
			if ( phase == 0 ) 
			{
				do {
					T U1 = (T)::rand() / RAND_MAX;
					T U2 = (T)::rand() / RAND_MAX;
					V1 = 2 * U1 - 1;
					V2 = 2 * U2 - 1;
					S = V1 * V1 + V2 * V2;
				} while(S >= 1 || S == 0);
				X = V1 * sqrt(-2 * ::log(S) / S);
			} else
				X = V2 * sqrt(-2 * ::log(S) / S);
			phase = 1 - phase;
			data[i]=X*rate;
		}
		return 0;
	}

	//普通随机分布
	template<typename T>
	int rand(T* data,int dataSize)
	{
		for(int i=0;i<dataSize;i++)
		{
			data[i]=(T)::rand() / RAND_MAX;
		}
		return 0;
	}

};
#endif