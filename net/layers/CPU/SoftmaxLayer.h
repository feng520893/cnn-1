#ifndef SOFTMAX_LAYER_CPU_H
#define SOFTMAX_LAYER_CPU_H
#include "LayerBaseCPU.h"

class CSoftmaxLayerCPU : public CLayerBaseCPU
{
public:
	CSoftmaxLayerCPU(){};
	~CSoftmaxLayerCPU(){freeMem();};
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int     backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
	int     getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms);

	int setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
	void   freeMem();
};

#endif