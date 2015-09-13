#ifndef SOFTMAX_LAYER_GPU_H
#define SOFTMAX_LAYER_GPU_H
#include"LayerBaseGPU.cuh"

class CSoftMaxLayerGPU : public CLayerGPU
{
public:
	CSoftMaxLayerGPU()
	{
		m_truthData=m_fullData=NULL;
	}
	~CSoftMaxLayerGPU()
	{
		freeMem();
	}
	int     initMem();
	void    freeMem();
	void    feedforward(double*srcData,DLparam& params);
	void    backpropagation(double*preDelta,DLparam& params);
	void    getGrad(double* srcData);
 	void    updateWeight(float mom,float alpha);
 	double  getCost(DLparam& params);

private:
	double* m_fullData;
	double* m_truthData;
};

#endif