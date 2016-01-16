#ifndef SIGMOID_LAYER_GPU_H
#define SIGMOID_LAYER_GPU_H
#include"LayerBaseGPU.cuh"
class CSigmoidLayerGPU : public CLayerBaseGPU
{
public:
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
};
#endif