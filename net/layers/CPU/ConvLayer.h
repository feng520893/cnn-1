#ifndef CONV_LAYER_CPU_H
#define CONV_LAYER_CPU_H
#include "LayerBaseCPU.h"

class CConvLayerCPU : public CLayerBaseCPU
{
public:
	int    setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
private:
	int       getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms);
};

#endif