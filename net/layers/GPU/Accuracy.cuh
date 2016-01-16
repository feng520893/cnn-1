#ifndef ACCURACY_GPU_H
#define ACCURACY_GPU_H
#include"LayerBaseGPU.cuh"

class CAccuracyGPU : public CLayerBaseGPU
{
public:
	int setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
private:
	Blob<precision> m_corrects;
};

#endif