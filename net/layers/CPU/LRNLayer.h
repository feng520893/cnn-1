#ifndef LRN_LAYER_CPU_H
#define LRN_LAYER_CPU_H
#include"LayerBaseCPU.h"
class CLRNLayerCPU : public CLayerBaseCPU
{
public:
	int    setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
private:
	int m_prePad;
	Blob<precision> m_scales;
	virtual int crossChannelForward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	virtual int crossChannelBack(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
};
#endif