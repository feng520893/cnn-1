#ifndef LAYERS_GPU_H
#define LAYERS_GPU_H
#include<cuda_runtime.h>
#include<stdio.h>
#include<vector>
#include<cmath>
#include<string>

#include"..\\LayerBase.h"

#define MAX_THREAD_NUM 32

#define GPU_FREE(x) if(x)cudaFree(x);x=NULL;

struct CLayerGPU : public CLayer
{

	CLayerGPU(){};

	~CLayerGPU(){freeMem();};

	void   setWeightValue(int index,double value);
	double getWeightValue(int inddex);

	void   setBiasValue(int index,double value);
	double getBiasValue(int inddex);

	void getWeightsGrad(double* CPUdata)
	{
		cudaError_t state=cudaMemcpy(CPUdata,m_weightGrad,sizeof(double)*m_weightLen,cudaMemcpyDeviceToHost);
		CUDA_ERROR(state);
	}

	void getBiasGrad(double* CPUdata)
	{
		cudaError_t state=cudaMemcpy(CPUdata,m_biasGrad,sizeof(double)*m_curNumFeature,cudaMemcpyDeviceToHost);
		CUDA_ERROR(state);
	}

	virtual void    feedforward(double* srcData,DLparam& params){};
	virtual void    backpropagation(double*preDelta,DLparam& params){};
	virtual void    getGrad(double* srcData){};
	virtual double  getCost(DLparam& params){return 0;};
	virtual double* getOutPut(){return NULL;};
	virtual double* getDelta(){return m_delta;};
	virtual void    updateWeight(float mom,float alpha){};

	int    save(FILE* fp);
	int    load(FILE*fp);

	virtual int    initMem();
	virtual void   freeMem();
};

/*静态混合编译会导致训练有问题
__device__ double activeFun(double src,int type);

__device__ double d_activeFun(double src,int type);*/

/**************************************************************** 
                          通用函数
*****************************************************************/

__global__ void g_weightAndBiasAdd(double* weights,double* weightGrads,
								   double* vec_weight,double* bias,
								   double* biasGrad,double* vec_bias,
								   int oneFeatureWeightSize,
								   float mom,float alpha);

// 获取权值的平方和
double getWeightCost(double* devWeight,unsigned int dataSize);


/*
	因为传入数据以行为主，而CUDA以列为主
	SO:z = x * y'==>z'=x'*y;
*/
void matrixMulTA(double * x,int rowsX,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ);

/*
	因为传入数据以行为主，而CUDA以列为主
	SO:z = x' * y==>z'=x*y';
*/
void matrixMulTB(double * x,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ);

/*
	因为传入数据以行为主，而CUDA以列为主
	SO:z = x * y==>z'=y*x;
*/
void matrixMul(double * x,int rowsX,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ);

/*全连层和softmax层共用函数*/
__global__ void fullWeightGrad(double* wgrad, double* weight, double lambda, int batch); 

__global__ void fullWeightGrad2(double* wgrad, double* weight,float* dropW,int weightLeng,double lambda, int batch); 

__global__ void fullBiasGrad( double* delta, double* grads,int batch);

__global__ void g_fullConnect(double* srcData,double* weight,
						   int inputNumFeature,double* fullData,
						   double* bias);
#endif