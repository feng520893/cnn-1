#ifndef SOFTMAX_LAYER_GPU_H
#define SOFTMAX_LAYER_GPU_H
#include"LayerBaseGPU.cuh"

class CSoftMaxLayerGPU : public CLayerBaseGPU
{
public:
	CSoftMaxLayerGPU(){}
	~CSoftMaxLayerGPU(){}
	
	int setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
	int       getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms);
};

#endif