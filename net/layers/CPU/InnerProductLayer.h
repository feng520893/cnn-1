#ifndef FULL_LAYER_CPU_H
#define FULL_LAYER_CPU_H
#include "LayerBaseCPU.h"

class CInnerProductLayerCPU : public CLayerBaseCPU
{
public:
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
	int       getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms);

	int    setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
};

#endif