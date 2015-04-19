#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include"LayerBase.cuh"

class CSoftMaxLayer : public CLayer
{
public:
	CSoftMaxLayer()
	{
		m_truthData=m_fullData=m_gpuTruthDelta=NULL;
	}
	~CSoftMaxLayer()
	{
		freeMem();
	}
	int     initMem();
	void    freeMem();
	int     feedforward(double** pred,double*srcData,bool bPred=false);
	double* backpropagation(std::vector<int>& labels);
	void    getGrad(double* srcData);
 	void    updateWeight(float mom,float alpha);
 	double  getCost(std::vector<int>& labels);

	void getWeightsGrad(double* CPUdata)
	{
		cudaMemcpy(CPUdata,m_weightGrad,sizeof(double)*m_inputNumFeature*m_curNumFeature,cudaMemcpyDeviceToHost);
	}

	void getBiasGrad(double* CPUdata)
	{
		cudaMemcpy(CPUdata,m_biasGrad,sizeof(double)*m_curNumFeature,cudaMemcpyDeviceToHost);
	}

private:
	double* m_fullData;
	double* m_truthData;
	double* m_gpuTruthDelta;
};

#endif