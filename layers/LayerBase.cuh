#ifndef LAYERS_H
#define LAYERS_H
#include<cuda_runtime.h>
#include<stdio.h>
#include<vector>
#include<cmath>

#include"..\\common\\common.h"

#define NL_NONE      0
#define NL_SOFT_PLUS 1 
#define NL_RELU      2 

#define MAX_THREAD_NUM 32

#define GPU_FREE(x) if(x)cudaFree(x);x=NULL;

struct CLayer
{
	short   m_inputNumFeature;
	short   m_curNumFeature;
	float   m_lambda;
	double* m_weight;
	double* m_bias;
	double* m_weightGrad;
	double* m_biasGrad;
	double* m_delta;
	double* m_vecWeight;
	double* m_vecBias;
	int     batch;
	int     m_weightLen;

	CLayer();
	~CLayer(){freeMem();};
	void   setWeightValue(int index,double value);
	double getWeightValue(int inddex);

	void   setBiasValue(int index,double value);
	double getBiasValue(int inddex);

	int    save(FILE* fp);
	int    load(FILE*fp);

	int    initMem();
	void   freeMem();
};

/*��̬��ϱ���ᵼ��ѵ��������
__device__ double activeFun(double src,int type);

__device__ double d_activeFun(double src,int type);*/

/**************************************************************** 
                          ͨ�ú���
*****************************************************************/

__global__ void g_weightAndBiasAdd(double* weights,double* weightGrads,
								   double* vec_weight,double* bias,
								   double* biasGrad,double* vec_bias,
								   int oneFeatureWeightSize,
								   float mom,float alpha);

// ��ȡȨֵ��ƽ����
double getWeightCost(double* devWeight,unsigned int dataSize);


/*
	��Ϊ������������Ϊ������CUDA����Ϊ��
	SO:z = x * y'==>z'=x'*y;
*/
void matrixMulTA(double * x,int rowsX,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ);

/*
	��Ϊ������������Ϊ������CUDA����Ϊ��
	SO:z = x' * y==>z'=x*y';
*/
void matrixMulTB(double * x,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ);

/*
	��Ϊ������������Ϊ������CUDA����Ϊ��
	SO:z = x * y==>z'=y*x;
*/
void matrixMul(double * x,int rowsX,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ);

/*ȫ�����softmax�㹲�ú���*/
__global__ void fullWeightGrad(double* wgrad, double* weight, double lambda, int batch); 

__global__ void fullWeightGrad2(double* wgrad, double* weight,float* dropW,int weightLeng,double lambda, int batch); 

__global__ void fullBiasGrad( double* delta, double* grads,int batch);

__global__ void g_fullConnect(double* srcData,double* weight,
						   int inputNumFeature,double* fullData,
						   double* bias);
#endif