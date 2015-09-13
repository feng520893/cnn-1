#ifndef LAYERS_H
#define LAYERS_H
#include<cuda_runtime.h>
#include<stdio.h>
#include<vector>
#include<cmath>
#include<string>

#include"..\\common\\common.h"
#include"..\\common\\dlError.h"

#define NL_NONE      0
#define NL_SOFT_PLUS 1 
#define NL_RELU      2

enum POOL_TYPE{MAX_POOL=0,AVG_POOL};

typedef double precision;

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
	std::string m_name;
	std::string m_inputName;

	CLayer():m_weightLen(0),m_curNumFeature(0)
	{
		m_vecBias=m_vecWeight=m_weight=m_bias=m_weightGrad=m_biasGrad=m_delta=NULL;
	};

	virtual void   setWeightValue(int index,double value)=0;
	virtual double getWeightValue(int inddex)=0;

	virtual void setBiasValue(int index,double value)=0;
	virtual double getBiasValue(int index)=0;

	virtual void getWeightsGrad(double* gradWeightData)=0;

	virtual void getBiasGrad(double* gradBiasData)=0;

	virtual int    save(FILE* fp)=0;
	virtual int    load(FILE*fp)=0;

	virtual int    initMem()=0;
	virtual void   freeMem()=0;

	virtual void    feedforward(double* srcData,DLparam& params){};
	virtual void    backpropagation(double*preDelta,DLparam& params){};
	virtual void    getGrad(double* srcData){};
	virtual double  getCost(DLparam& params){return 0;};
	virtual double* getOutPut(){return NULL;};
	virtual double* getDelta(){return m_delta;};
	virtual void    updateWeight(float mom,float alpha){};
};
#endif