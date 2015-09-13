#ifndef FULL_LAYER_CPU_H
#define FULL_LAYER_CPU_H
#include "LayerBaseCPU.h"
#include"..\FullLayerBase.h"

class CFullLayerCPU : public CLayerBaseCPU,public CFullLayerBase
{
public:
	CFullLayerCPU(){m_fullData=NULL;};
	~CFullLayerCPU(){freeMem();};
	void    feedforward(double* srcData,DLparam& params);
	void    backpropagation(double*preDelta,DLparam& params);
	void    getGrad(double* srcData);
	double  getCost(DLparam& params);
	double* getOutPut(){return m_fullData;};

	int    initMem();
	void   freeMem();
};

#endif