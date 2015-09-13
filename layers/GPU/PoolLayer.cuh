#ifndef POOLLAYER_GPU_H
#define POOLLAYER_GPU_H
#include"LayerBaseGPU.cuh"
#include"../poolLayerBase.h"

class CPoolLayerGPU : public CLayerGPU,public CPoolLayerBase
{
public:
	CPoolLayerGPU(){}
	~CPoolLayerGPU()
	{
		freeMem();
	}
	int     initMem();
	void    freeMem();

	void    feedforward(double* srcData,DLparam& params);
	void    backpropagation(double* nexDetals,DLparam& params);

	double* getOutPut(){return m_poolData;};
};

#endif