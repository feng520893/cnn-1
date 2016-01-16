#ifndef DROP_OUT_LAYER_GPU_H
#define DROP_OUT_LAYER_GPU_H
#include"LayerBaseGPU.cuh"
class CDropoutLayerGPU : public CLayerBaseGPU
{
public:
	CDropoutLayerGPU()
	{
		m_param.dropoutParam.dropoutRate=0.0;
		addBlobsToShareOtherLayer(&m_mask);
	};

	int setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
	Blob<precision> m_mask;
};
#endif