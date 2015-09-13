#ifndef CONVLAYER_GPU_H
#define CONVLAYER_GPU_H
#include"LayerBaseGPU.cuh"
#include "../ConvLayerBase.h"

class CConvLayerGPU : public CLayerGPU, public CConvLayerBase
{
public:
	CConvLayerGPU()
	{
		m_deltaSum=m_convData=m_convNoActiveData=m_weightTmp=NULL;
	}
	~CConvLayerGPU(){freeMem();};
	void    feedforward(double* srcData,DLparam& params);
	void    backpropagation(double*preDelta,DLparam& params);
	void    getGrad(double* srcData);
 	void    updateWeight(float mom,float alpha);
 	double  getCost(DLparam& params);
	int     initMem();
	void    freeMem();
	
	virtual double* getOutPut(){return m_convData;};

private:
	double* m_weightTmp;
	double* m_deltaSum;
};

#endif