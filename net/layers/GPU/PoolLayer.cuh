#ifndef POOLLAYER_GPU_H
#define POOLLAYER_GPU_H
#include"LayerBaseGPU.cuh"

class CPoolLayerGPU : public CLayerBaseGPU
{
public:
	CPoolLayerGPU(){}
	CPoolLayerGPU(LayerParam& param)
	{
		addBlobsToShareOtherLayer(&m_maxIndexData);
	}

	int setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);

	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);

	Blob<precision> m_maxIndexData;
};

#endif