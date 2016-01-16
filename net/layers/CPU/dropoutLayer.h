#ifndef DROP_OUT_LAYER_CPU_H
#define DROP_OUT_LAYER_CPU_H
#include"LayerBaseCPU.h"
class CDropoutLayerCPU : public CLayerBaseCPU
{
public:
	CDropoutLayerCPU()
	{
		m_param.dropoutParam.dropoutRate=0.0;
		addBlobsToShareOtherLayer(&m_mask);
	};

	int setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
private:
	Blob<precision> m_mask;
};
#endif