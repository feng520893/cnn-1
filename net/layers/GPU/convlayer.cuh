#ifndef CONVLAYER_GPU_H
#define CONVLAYER_GPU_H
#include"LayerBaseGPU.cuh"

class CConvLayerGPU : public CLayerBaseGPU
{
public:
	CConvLayerGPU()
	{
		m_deltaSum=m_weightTmp=NULL;
	}
	~CConvLayerGPU(){freeMem();};
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
	int       backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms);
	int       setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);

private:
	double* m_weightTmp;
	double* m_deltaSum;
	
	void      freeMem();
	int       getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms);
};

#endif