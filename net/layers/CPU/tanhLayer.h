#ifndef TANH_LAYER_CPU_H
#define TANH_LAYER_CPU_H
#include"LayerBaseCPU.h"
class CTanhLayerCPU : public CLayerBaseCPU
{
public:
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
};
#endif