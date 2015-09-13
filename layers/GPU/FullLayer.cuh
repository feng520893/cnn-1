#ifndef FULLLAYER_GPU_H
#define FULLLAYER_GPU_H
#include"LayerBaseGPU.cuh"
#include"..\FullLayerBase.h"

#include <curand.h>

class CFullLayerGPU : public CLayerGPU,public CFullLayerBase
{
public:
	CFullLayerGPU()
	{
		m_afterDropWeight=m_fullData=m_fullNoActiveData=NULL;
		m_dropRate=NULL;
		m_dropRate=0.0;
		m_hGen=NULL;
	}
	~CFullLayerGPU()
	{
		freeMem();
	}
	int     initMem();
	void    freeMem();

	void    feedforward(double* srcData,DLparam& params);
	void    backpropagation(double* preDetals,DLparam& params);
	void    getGrad(double* srcData);
 	void    updateWeight(float mom,float alpha);
 	double  getCost(DLparam& params);

	virtual double* getOutPut(){return m_fullData;};
	
protected:
	curandGenerator_t m_hGen;
};

#endif