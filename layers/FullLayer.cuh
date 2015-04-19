#ifndef FULLLAYER_H
#define FULLLAYER_H
#include"LayerBase.cuh"

#include <curand.h>

class CFullLayer : public CLayer
{
public:
	CFullLayer()
	{
		m_afterDropWeight=m_fullData=m_fullNoActiveData=m_fullDelta=NULL;
		m_dropRate=NULL;
		m_rate=0.0;
		m_hGen=NULL;
	}
	~CFullLayer()
	{
		freeMem();
	}
	int     initMem();
	void    freeMem();

	void    feedforward(double* srcData,int activeType,bool bPred);
	int     backpropagation(double* nexDetals,int activeType);
	void    getGrad(double* srcData);
 	void    updateWeight(float mom,float alpha);
 	double  getCost();

	void getWeightsGrad(double* CPUdata)
	{
		cudaMemcpy(CPUdata,m_weightGrad,sizeof(double)*m_inputNumFeature*m_curNumFeature,cudaMemcpyDeviceToHost);
	}

	void getBiasGrad(double* CPUdata)
	{
		cudaMemcpy(CPUdata,m_biasGrad,sizeof(double)*m_curNumFeature,cudaMemcpyDeviceToHost);
	}

	double* m_fullData;
	float   m_rate;
protected:
	double* m_fullNoActiveData;
	
	double* m_afterDropWeight;
	
	float* m_dropRate;
	double* m_fullDelta;

	curandGenerator_t m_hGen;
};

#endif