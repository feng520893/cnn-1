#ifndef SOFTMAX_LAYER_CPU_H
#define SOFTMAX_LAYER_CPU_H
#include "LayerBaseCPU.h"

class CSoftmaxLayerCPU : public CLayerBaseCPU
{
public:
	CSoftmaxLayerCPU(){};
	~CSoftmaxLayerCPU(){freeMem();};
	void    feedforward(double* srcData,DLparam& params);
	void    backpropagation(double*preDelta,DLparam& params);
	void    getGrad(double* srcData);
	double  getCost(DLparam& params);

	int    initMem();
	void   freeMem();

private:
	precision* m_predDatas;
	precision* m_trueDatas;
};

#endif