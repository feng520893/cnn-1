#ifndef CONV_LAYER_CPU_H
#define CONV_LAYER_CPU_H
#include "LayerBaseCPU.h"
#include "../ConvLayerBase.h"

class CConvLayerCPU : public CLayerBaseCPU, public CConvLayerBase
{
public:
	void    feedforward(double* srcData,DLparam& params);
	void    backpropagation(double*preDelta,DLparam& params);
	void    getGrad(double* srcData);
	double  getCost(DLparam& params);
	double* getOutPut(){return m_convData;};

	int    initMem();
	void   freeMem();
};

#endif