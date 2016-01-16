#ifndef LAYERS_GPU_H
#define LAYERS_GPU_H
#include<stdio.h>
#include<vector>
#include<cmath>
#include<string>

#include"..\\LayerBase.h"

#define GPU_FREE(x) if(x)cudaFree(x);x=NULL;

struct CLayerBaseGPU : public CLayer
{

	CLayerBaseGPU(){};

	~CLayerBaseGPU(){freeMem();};

	int setup(std::vector<Blob<precision>*>& input,std::vector<Blob<precision>*>& output);

	void      setWeightValue(int index,precision value);
	precision getWeightValue(int index);

	void      setBiasValue(int index,precision value);
	precision getBiasValue(int index);

	void getWeightsGrad(precision* CPUdata)
	{
		cudaError_t state=cudaMemcpy(CPUdata,m_weightGrad,sizeof(precision)*m_weightLen,cudaMemcpyDeviceToHost);
		CUDA_ERROR(state);
	}

	void getBiasGrad(precision* CPUdata)
	{
		cudaError_t state=cudaMemcpy(CPUdata,m_biasGrad,sizeof(precision)*m_param.curNumFeature,cudaMemcpyDeviceToHost);
		CUDA_ERROR(state);
	}

	int    save(FILE* fp);
	int    load(FILE*fp);

	virtual void   freeMem();
	virtual int    updateWeight(float mom,float baseLR);
protected:
	void computeBiasGrad(precision* delta, precision* grads,int batch);
};

/*静态混合编译会导致训练有问题
__device__ double activeFun(double src,int type);

__device__ double d_activeFun(double src,int type);*/

/**************************************************************** 
                          通用函数
*****************************************************************/

__global__ void g_weightAndBiasAdd(precision* weights,precision* weightGrads,
								   precision* vec_weight,precision* bias,
								   precision* biasGrad,precision* vec_bias,
								   int oneFeatureWeightSize,
								   float mom,float weightsLR,
								   float biasLR);

// 获取权值的平方和
double getWeightCost(precision* devWeight,unsigned int dataSize);

/*全连层和softmax层共用函数*/
__global__ void fullWeightGrad(precision* wgrad, precision* weight,int weightLeng,float lambda, int batch); 

__global__ void g_fullConnect(precision* srcData,precision* weight,
						   int inputNumFeature,precision* fullData,
						   precision* bias);
#endif