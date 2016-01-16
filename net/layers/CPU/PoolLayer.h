#ifndef POOL_LAYER_CPU_H
#define POOL_LAYER_CPU_H
#include "LayerBaseCPU.h"

class CPoolLayerCPU : public CLayerBaseCPU
{
public:
	CPoolLayerCPU(){};
	CPoolLayerCPU(LayerParam& param)
	{
		addBlobsToShareOtherLayer(&m_maxIndexData);
	}

	int    setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);

	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
private:
	Blob<precision> m_maxIndexData;
};

#endif