#ifndef POOL_LAYER_CPU_H
#define POOL_LAYER_CPU_H
#include "LayerBaseCPU.h"
#include"../poolLayerBase.h"

class CPoolLayerCPU : public CLayerBaseCPU,public CPoolLayerBase
{
public:
	CPoolLayerCPU(){};
	~CPoolLayerCPU(){freeMem();};
	void    feedforward(double* srcData,DLparam& params);
	void    backpropagation(double*preDelta,DLparam& params);
	double* getOutPut(){return m_poolData;};

	int    initMem();
	void   freeMem();
};

#endif