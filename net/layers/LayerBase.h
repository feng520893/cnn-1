#ifndef LAYERS_H
#define LAYERS_H
#include<stdio.h>
#include<vector>
#include<cmath>
#include<string>

#include"../blob.h"
#include"../LayerParam.h"
#include"../netDefine.h"
#include"../../common/tools.h"

struct CLayer
{
	LayerParam m_param;
	Blob<precision> m_weight;
	Blob<precision> m_bias;
	precision* m_weightGrad;
	precision* m_biasGrad;
	precision* m_vecWeight;
	precision* m_vecBias;
	int        m_weightLen;
	std::vector<Blob<precision>*> m_layerBlobs;

	CLayer():m_weightLen(0)
	{
		m_vecBias=m_vecWeight=m_weightGrad=m_biasGrad=NULL;
	};
	CLayer(LayerParam& param):m_param(param)
	{
	};

	virtual int setup(std::vector<Blob<precision>*>& input,std::vector<Blob<precision>*>& output)
	{
		addBlobsToShareOtherLayer(&m_weight);
		addBlobsToShareOtherLayer(&m_bias);
		return NET_SUCCESS;
	};
	virtual void   freeMem(){};

	//让不同网络同样的layer共享内存数据
	void addBlobsToShareOtherLayer(Blob<precision>* blob)
	{
		m_layerBlobs.push_back(blob);
	}

	virtual void   setWeightValue(int index,double value){};
	virtual double getWeightValue(int inddex){return 0;};
	virtual void   setBiasValue(int index,double value){};
	virtual double getBiasValue(int index){return 0;};
	virtual void   getWeightsGrad(double* gradWeightData){};
	virtual void   getBiasGrad(double* gradBiasData){};

	virtual int    save(FILE* fp){return 0;};
	virtual int    load(FILE*fp){return 0;};

	virtual precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops){return 0;};
	virtual int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms){return 0;};

	virtual int    updateWeight(float mom,float baseLR){return 0;};
};
#endif